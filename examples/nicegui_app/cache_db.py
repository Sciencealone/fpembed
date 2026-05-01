"""SQLite-backed cache for molecular descriptors, fingerprints, and images.

Stripped version for NiceGUI app: no optimization_runs table,
no save_run_state/load_run_state, no CSV migration.
"""

import logging
import os
import sqlite3

import numpy as np
import pandas as pd

from fpembed import parse_smiles
from chemistry import custom_descriptors, smiles_to_image_data_uri

logger = logging.getLogger(__name__)


class SQLiteCacheDB:
    """Unified SQLite cache for molecular descriptors, fingerprints, and images."""

    def __init__(self, db_path: str, descriptor_list: list[str]):
        self.db_path = db_path
        self.descriptor_list = descriptor_list
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._init_tables()
        except Exception as e:
            logger.warning("Could not open cache DB at %s (%s). Recreating.", db_path, e)
            try:
                if self.conn:
                    self.conn.close()
                if os.path.exists(db_path):
                    os.remove(db_path)
                self.conn = sqlite3.connect(db_path, check_same_thread=False)
                self.conn.execute("PRAGMA journal_mode=WAL")
                self._init_tables()
            except Exception as e2:
                logger.error("Failed to create new cache DB: %s", e2)
                raise

    def _init_tables(self):
        """Create fingerprints, descriptors, and images tables if needed."""
        cur = self.conn.cursor()
        # Schema mismatch detection: drop old fingerprints table if present
        cur.execute("PRAGMA table_info(fingerprints)")
        columns = {row[1] for row in cur.fetchall()}
        if columns and ("fp_type" not in columns or "fp_params_hash" not in columns):
            logger.warning("Old fingerprints schema detected. Dropping and recreating.")
            cur.execute("DROP TABLE fingerprints")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                canonical_smiles TEXT NOT NULL, fp_type TEXT NOT NULL,
                fp_size INTEGER NOT NULL, compression INTEGER NOT NULL,
                fp_params_hash TEXT NOT NULL, fingerprint_blob BLOB NOT NULL,
                PRIMARY KEY (canonical_smiles, fp_type, fp_size, compression, fp_params_hash)
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS descriptors (
                canonical_smiles TEXT NOT NULL, descriptor_name TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (canonical_smiles, descriptor_name)
            )""")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_descriptors_smiles
                ON descriptors (canonical_smiles)""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                canonical_smiles TEXT PRIMARY KEY, image_data_uri TEXT NOT NULL
            )""")
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    # -- Fingerprint methods -------------------------------------------

    def get_fingerprint(self, canonical_smiles, fp_type, fp_size,
                        compression, fp_params_hash):
        """Retrieve a single cached fingerprint, or None on cache miss."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT fingerprint_blob FROM fingerprints "
            "WHERE canonical_smiles=? AND fp_type=? AND fp_size=? "
            "AND compression=? AND fp_params_hash=?",
            (canonical_smiles, fp_type, fp_size, compression, fp_params_hash),
        )
        row = cur.fetchone()
        return np.frombuffer(row[0], dtype=np.float64).copy() if row else None

    def store_fingerprint(self, canonical_smiles, fp_type, fp_size,
                          compression, fp_params_hash, fingerprint):
        """Store a fingerprint blob (INSERT OR IGNORE)."""
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO fingerprints "
                "(canonical_smiles, fp_type, fp_size, compression, "
                "fp_params_hash, fingerprint_blob) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    canonical_smiles, fp_type, fp_size, compression,
                    fp_params_hash,
                    fingerprint.astype(np.float64).tobytes(),
                ),
            )
            self.conn.commit()
        except Exception as e:
            logger.warning("Failed to store fingerprint for %s: %s", canonical_smiles, e)

    def get_fingerprints_batch(self, smiles_list, generator):
        """Retrieve fingerprints for a batch; compute and cache any missing."""
        fp_type = generator.fp_type
        fp_size = generator.fp_size
        compression = generator.compression
        p_hash = generator.params_hash

        results = [None] * len(smiles_list)
        missing_indices = []

        for i, smi in enumerate(smiles_list):
            cached = self.get_fingerprint(smi, fp_type, fp_size, compression, p_hash)
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)

        if missing_indices:
            for idx in missing_indices:
                smi = smiles_list[idx]
                fp_vec = generator.GetFingerprintFromSmiles(smi)
                if fp_vec is None:
                    expected_len = (
                        fp_size // compression if compression > 0 else fp_size
                    )
                    fp_vec = np.zeros(expected_len, dtype=np.float64)
                results[idx] = fp_vec
                self.store_fingerprint(smi, fp_type, fp_size, compression, p_hash, fp_vec)

        return np.array(results, dtype=np.float64)

    # -- Descriptor methods ---------------------------------------------

    def get_descriptors(self, canonical_smiles):
        """Retrieve all cached descriptors for a molecule, or None on miss."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT descriptor_name, value FROM descriptors WHERE canonical_smiles=?",
            (canonical_smiles,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        cached = {name: val for name, val in rows}
        if all(d in cached for d in self.descriptor_list):
            return {d: cached[d] for d in self.descriptor_list}
        return None

    def store_descriptors(self, canonical_smiles, descriptors):
        """Store descriptor name/value pairs (INSERT OR IGNORE)."""
        try:
            self.conn.executemany(
                "INSERT OR IGNORE INTO descriptors "
                "(canonical_smiles, descriptor_name, value) VALUES (?, ?, ?)",
                [(canonical_smiles, name, val) for name, val in descriptors.items()],
            )
            self.conn.commit()
        except Exception as e:
            logger.warning("Failed to store descriptors for %s: %s", canonical_smiles, e)

    def get_descriptors_batch(self, smiles_list):
        """Retrieve descriptors for a batch; compute and cache any missing."""
        all_descriptors = [None] * len(smiles_list)
        missing_indices = []

        for i, smi in enumerate(smiles_list):
            cached = self.get_descriptors(smi)
            if cached is not None:
                all_descriptors[i] = cached
            else:
                missing_indices.append(i)

        for idx in missing_indices:
            smi = smiles_list[idx]
            mol = parse_smiles(smi)
            desc = (
                custom_descriptors(mol, self.descriptor_list, missingVal=0)
                if mol is not None
                else {d: 0.0 for d in self.descriptor_list}
            )
            all_descriptors[idx] = desc
            self.store_descriptors(smi, desc)

        return pd.DataFrame(all_descriptors, columns=self.descriptor_list).astype(np.float64)

    # -- Image caching methods ------------------------------------------

    def get_image(self, canonical_smiles):
        """Retrieve a cached image data URI, or None on miss."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT image_data_uri FROM images WHERE canonical_smiles=?",
            (canonical_smiles,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def store_image(self, canonical_smiles, image_data_uri):
        """Store an image data URI (INSERT OR IGNORE)."""
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO images (canonical_smiles, image_data_uri) VALUES (?, ?)",
                (canonical_smiles, image_data_uri),
            )
            self.conn.commit()
        except Exception as e:
            logger.warning("Failed to store image for %s: %s", canonical_smiles, e)

    def get_images_batch(self, smiles_list):
        """Retrieve image data URIs for a batch; compute and cache missing."""
        results = [None] * len(smiles_list)
        missing_indices = []

        for i, smi in enumerate(smiles_list):
            cached = self.get_image(smi)
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)

        for idx in missing_indices:
            img = smiles_to_image_data_uri(smiles_list[idx])
            results[idx] = img
            if img != "N/A":
                self.store_image(smiles_list[idx], img)

        return results
