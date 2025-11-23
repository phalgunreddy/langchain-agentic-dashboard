#!/usr/bin/env python3
"""Ingest preprocessed JSW data into the system"""

import sys
sys.path.append('../1_core')

from smart_preprocessor import JSWEnergyPreprocessor
from embedding_store import EmbeddingStore
import os

print("\n🚀 INGESTING PREPROCESSED JSW ENERGY DATA")
print("="*60)

# Clean old data
print("\n🗑️  Clearing old database...")
data_dir = '../4_data/data_prototype'
if os.path.exists(f'{data_dir}/indexes/faiss.index'):
    os.remove(f'{data_dir}/indexes/faiss.index')
if os.path.exists(f'{data_dir}/indexes/id_map.json'):
    os.remove(f'{data_dir}/indexes/id_map.json')
if os.path.exists(f'{data_dir}/metadata.db'):
    os.remove(f'{data_dir}/metadata.db')

# Process and ingest
print("\n📊 Processing Excel file...")
processor = JSWEnergyPreprocessor()
docs = processor.process_file("../4_data/Energy Consumption Daily Report MHS Ele - Copy.xlsx")

print(f"\n💾 Adding {len(docs)} documents to embedding store...")
es = EmbeddingStore()
es.add_documents(docs)

print(f"\n✅ COMPLETE! Documents in store: {es.faiss_index.ntotal}")
print("\n📝 Sample documents:")
for i, doc in enumerate(docs[:3], 1):
    print(f"\n{i}. {doc.doc_type}:")
    print(f"   {doc.content[:120]}...")

print(f"\n🔍 Testing search...")
results = es.search("total plant consumption", k=3)
print(f"\nTop result: {results[0]['content'][:150]}...")
