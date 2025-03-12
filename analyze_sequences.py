#!/usr/bin/env python

import os
import logging
import numpy as np
import torch
from Bio import SeqIO
from typing import List, Tuple, Dict

from interplm.sae.inference import load_sae_from_hf, get_sae_feats_in_batches
from interplm.esm.embed import embed_list_of_prot_seqs

import script_args

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sequences(path: str) -> List[Tuple[str, str]]:
    """load protein sequences from a single fasta file."""
    sequences = []
    for record in SeqIO.parse(path, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def extract_features(sequences: List[Tuple[str, str]],
                     plm_model: str,
                     plm_layer: int,
                     device: str,
                     batch_size: int,
                     max_seq_length: int) -> Dict[str, np.ndarray]:
    """extract features from protein sequences using a pretrained sae"""
    logger.info(f"Loading SAE model for {plm_model} layer {plm_layer}...")

    esm_model_map = {
        "esm2-8m": "esm2_t6_8M_UR50D",
        "esm2-650m": "esm2_t33_650M_UR50D",
    }
    esm_model_name = esm_model_map[plm_model]

    sae = load_sae_from_hf(plm_model=plm_model, plm_layer=plm_layer)
    sae.to(device)

    results = {}

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_ids, batch_seqs = zip(*batch)
        batch_seqs = [seq[:max_seq_length] for seq in batch_seqs]

        logger.info(f"Embedding batch {i // batch_size + 1}/{len(sequences) // batch_size + 1}...")
        embeddings_list = embed_list_of_prot_seqs(
            protein_seq_list=batch_seqs,
            esm_model_name=esm_model_name,
            layer=plm_layer,
            device=device
        )

        logger.info(f"Extracting SAE features for batch {i // batch_size + 1}...")
        for j, emb in enumerate(embeddings_list):
            seq_id = batch_ids[j]
            features = get_sae_feats_in_batches(
                sae=sae,
                device=device,
                esm_embds=emb,
                chunk_size=1024
            )
            results[seq_id] = features.cpu().numpy()

    return results

def main():
    if not os.path.exists(script_args.GROUP1_PATH):
        raise ValueError(f"Group 1 path does not exist: {script_args.GROUP1_PATH}")

    if not os.path.exists(script_args.GROUP2_PATH):
        raise ValueError(f"Group 2 path does not exist: {script_args.GROUP2_PATH}")

    os.makedirs(script_args.OUTPUT_DIR, exist_ok=True)

    device = "cuda" if script_args.USE_GPU and torch.cuda.is_available() else "cpu"
    if script_args.USE_GPU and not torch.cuda.is_available():
        logger.warning("GPU requested but not available. Using CPU instead.")

    logger.info(f"Using device: {device}")

    logger.info("Loading sequences from Group 1...")
    group1_sequences = load_sequences(script_args.GROUP1_PATH)
    logger.info(f"Loaded {len(group1_sequences)} sequences from Group 1")

    logger.info("Loading sequences from Group 2...")
    group2_sequences = load_sequences(script_args.GROUP2_PATH)
    logger.info(f"Loaded {len(group2_sequences)} sequences from Group 2")

    logger.info("Extracting features for Group 1...")
    group1_features = extract_features(
        sequences=group1_sequences,
        plm_model=script_args.PLM_MODEL,
        plm_layer=script_args.PLM_LAYER,
        device=device,
        batch_size=script_args.BATCH_SIZE,
        max_seq_length=script_args.MAX_SEQ_LENGTH
    )

    logger.info("Extracting features for Group 2...")
    group2_features = extract_features(
        sequences=group2_sequences,
        plm_model=script_args.PLM_MODEL,
        plm_layer=script_args.PLM_LAYER,
        device=device,
        batch_size=script_args.BATCH_SIZE,
        max_seq_length=script_args.MAX_SEQ_LENGTH
    )

    # Save as .npy files
    group1_npy_path = os.path.join(script_args.OUTPUT_DIR, "group1_features.npy")
    group2_npy_path = os.path.join(script_args.OUTPUT_DIR, "group2_features.npy")

    np.save(group1_npy_path, group1_features)
    np.save(group2_npy_path, group2_features)

    logger.info(f"Group 1 features saved to: {group1_npy_path}")
    logger.info(f"Group 2 features saved to: {group2_npy_path}")

    logger.info(f"Analysis complete! Results saved to {script_args.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
