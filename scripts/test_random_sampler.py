"""Example script demonstrating RandomQuerySampler usage.

This script shows how to use the RandomQuerySampler to randomly sample
a document from a ChromaDB collection and query for similar documents.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.stores.chroma_store import ChromaStore
from src.data_pipeline.samplers import RandomQuerySampler


def print_results(results, max_display=3):
    """Print query results in a readable format.

    Args:
        results: ChromaDB query results dictionary
        max_display: Maximum number of documents to display in detail
    """
    ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    total_docs = len(ids)
    print(f"\n{'='*80}")
    print(f"Query Results Summary")
    print(f"{'='*80}")
    print(f"Total documents returned: {total_docs}")
    print(f"  - Query document: 1 (distance ≈0)")
    print(f"  - Similar documents: {total_docs - 1}")
    print()

    # Display detailed results
    for i in range(min(max_display, total_docs)):
        doc_type = "QUERY DOCUMENT" if i == 0 else f"SIMILAR DOCUMENT #{i}"
        print(f"\n{'-'*80}")
        print(f"{doc_type}")
        print(f"{'-'*80}")
        print(f"ID: {ids[i]}")
        print(f"Distance: {distances[i]:.6f}")
        print(f"Metadata: {metadatas[i]}")
        print(f"\nDocument preview (first 200 chars):")
        doc_preview = documents[i][:200] + "..." if len(documents[i]) > 200 else documents[i]
        print(f"{doc_preview}")

    if total_docs > max_display:
        print(f"\n... and {total_docs - max_display} more documents")

    print(f"\n{'='*80}\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Random Query Sampler Example")
    parser.add_argument(
        "--collection", 
        type=str, 
        default="example_collection",
        help="Name of the ChromaDB collection to sample from (default: example_collection)"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=5,
        help="Number of similar documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--persist-dir", 
        type=str, 
        default="./chroma_db",
        help="Path to ChromaDB persistence directory (default: ./chroma_db)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def main():
    """Main function demonstrating RandomQuerySampler usage."""
    
    # Parse arguments
    args = parse_arguments()

    # Configuration
    PERSIST_DIR = args.persist_dir
    COLLECTION_NAME = args.collection
    N_RESULTS = args.n_results
    RANDOM_SEED = args.seed

    print("="*80)
    print("RandomQuerySampler Example")
    print("="*80)
    print(f"Configuration:")
    print(f"  - ChromaDB directory: {PERSIST_DIR}")
    print(f"  - Collection: {COLLECTION_NAME}")
    print(f"  - N results: {N_RESULTS}")
    print(f"  - Random seed: {RANDOM_SEED}")
    print("="*80)

    try:
        # Initialize ChromaStore
        print("\n[1/3] Initializing ChromaStore...")
        chroma_store = ChromaStore(persist_directory=PERSIST_DIR)
        print("✓ ChromaStore initialized")

        # List available collections
        collections = chroma_store.list_collections()
        print(f"✓ Available collections: {', '.join(collections)}")

        if COLLECTION_NAME not in collections:
            print(f"\n✗ Error: Collection '{COLLECTION_NAME}' not found!")
            print(f"Available collections: {', '.join(collections)}")
            return

        # Create RandomQuerySampler
        print(f"\n[2/3] Creating RandomQuerySampler (seed={RANDOM_SEED})...")
        sampler = RandomQuerySampler(chroma_store, random_seed=RANDOM_SEED)
        print("✓ RandomQuerySampler created")

        # Sample and query
        print(f"\n[3/3] Sampling from collection '{COLLECTION_NAME}' and querying...")
        results = sampler.sample_and_query(
            collection_name=COLLECTION_NAME,
            n_results=N_RESULTS
        )
        print(f"✓ Query completed successfully!")

        # Print results
        print_results(results, max_display=3)

        # Additional examples
        print("\n" + "="*80)
        print("Additional Examples")
        print("="*80)

        # Example 1: Different n_results
        print(f"\n[Example 1] Query with different n_results (n=2) using --collection {COLLECTION_NAME}...")
        results_small = sampler.sample_and_query(COLLECTION_NAME, n_results=2)
        print(f"✓ Returned {len(results_small['ids'][0])} documents (1 query + 2 similar)")

        # Example 2: Without random seed (truly random)
        print("\n[Example 2] Query without random seed (non-reproducible)...")
        sampler_random = RandomQuerySampler(chroma_store, random_seed=None)
        results_random = sampler_random.sample_and_query(COLLECTION_NAME, n_results=3)
        print(f"✓ Query document ID: {results_random['ids'][0][0]}")

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
