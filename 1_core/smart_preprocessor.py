#!/usr/bin/env python3
"""
Smart Preprocessor for JSW Energy Report
Based on manual analysis - extracts clean, labeled data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import uuid
import sys
sys.path.append('/Users/phalgun/Desktop/langchain-agentic-dashboard')
from ingestion_pipeline import Document

class JSWEnergyPreprocessor:
    """Intelligent preprocessor based on manual file analysis"""
    
    def __init__(self):
        self.monthly_sheets = [
            'JULY24', 'JUN24', 'APR24', 'MAR24', 'FEB24', 
            'JAN24', 'DEC23', 'NOV23', 'OCT23', 'SEPT23'
        ]
        self.skip_sheets = ['Dashboard', 'DATA SHEET', 'Sheet1', 'Environment', 'MAY24']
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process JSW Energy Excel with intelligent preprocessing"""
        
        print("\n" + "="*60)
        print("🧠 SMART JSW ENERGY PREPROCESSOR")
        print("="*60)
        
        documents = []
        file_id = str(uuid.uuid4())
        
        all_feeders = {}  # feeder_name -> {dates: {date: consumption}}
        daily_totals = {}  # date -> total_consumption
        monthly_totals = {}  # month -> total_consumption
        
        for sheet_name in self.monthly_sheets:
            try:
                print(f"\n📄 Processing: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
                
                if df.empty or len(df.columns) < 3:
                    print(f"   ⚠️  Skipping (empty or malformed)")
                    continue
                
                feeder_col = df.columns[0]
                swb_col = df.columns[1]
                
                # Find date columns
                date_cols = []
                for i, col in enumerate(df.columns[2:], start=2):
                    col_str = str(col)
                    if 'Difference' not in col_str and 'difference' not in col_str.lower():
                        try:
                            # Try to parse as date
                            if isinstance(col, (pd.Timestamp, datetime)):
                                date_cols.append((i, col))
                            elif isinstance(col, str) and any(c.isdigit() for c in col):
                                date_cols.append((i, col))
                        except:
                            pass
                
                print(f"   Found {len(date_cols)} dates, {len(df)} feeders")
                
                # Process each feeder
                for idx, row in df.iterrows():
                    feeder = str(row[feeder_col]).strip() if pd.notna(row[feeder_col]) else ""
                    swb = str(row[swb_col]).strip() if pd.notna(row[swb_col]) else "Unknown"
                    
                    # Skip invalid rows
                    if not feeder or feeder == "" or 'SWITCH BOARD' in feeder.upper():
                        continue
                    
                    # Create unique feeder key to handle duplicates (e.g. multiple I/C Panels)
                    feeder_key = f"{feeder} ({swb})"
                    
                    # Extract readings for each date
                    for col_idx, date in date_cols:
                        # We want the CONSUMPTION (Difference), which is usually in the next column
                        if col_idx + 1 >= len(row):
                            continue
                        
                        # CRITICAL FIX: Verify the next column is actually a "Difference" column
                        # If the next column is another Date, then the current column is just a start reading, not a consumption entry.
                        next_col_name = str(df.columns[col_idx + 1])
                        if "diff" not in next_col_name.lower() and "difference" not in next_col_name.lower():
                            continue

                        consumption_val = row.iloc[col_idx + 1] # The Difference Column
                        
                        if pd.isna(consumption_val) or consumption_val == 0:
                            continue
                        
                        try:
                            reading_val = float(consumption_val)
                        except:
                            continue
                        
                        # Parse date
                        if isinstance(date, (pd.Timestamp, datetime)):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)
                        
                        # Store feeder data
                        # Initialize feeder entry if not exists
                        if feeder_key not in all_feeders:
                            all_feeders[feeder_key] = {'name': feeder, 'swb': swb, 'readings': {}, 'last_diff': 0}

                        # OUTLIER CHECK & ACCUMULATION
                        daily_diff = reading_val
                        
                        # 0. Sanity Check: Consumption cannot be negative
                        if daily_diff < 0:
                            continue

                        is_outlier = False
                        
                        # 1. Hard limit (Monthly Total check)
                        if daily_diff > 500000:
                            is_outlier = True
                        
                        # 2. Dynamic check (Spike check)
                        if 'last_diff' in all_feeders[feeder_key]:
                            last_diff = all_feeders[feeder_key]['last_diff']
                            if last_diff > 0 and daily_diff > (last_diff * 10):
                                is_outlier = True
                                print(f"   ⚠️  Spike detected for {feeder_key}: {daily_diff} (prev diff: {last_diff}) on {date_str}")
                        
                        if is_outlier:
                            continue

                        # VALID READING: Store it now
                        all_feeders[feeder_key]['readings'][date_str] = daily_diff
                        all_feeders[feeder_key]['last_diff'] = daily_diff

                        # Add to daily totals
                        if date_str not in daily_totals:
                            daily_totals[date_str] = 0
                        daily_totals[date_str] += daily_diff
                        
                        # Add to monthly totals
                        month = sheet_name
                        if month not in monthly_totals:
                            monthly_totals[month] = 0
                        monthly_totals[month] += daily_diff
                        
                
                print(f"   ✅ Extracted data for {len([f for f in all_feeders if all_feeders[f]['readings']])} feeders")
            
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        # Create documents
        print(f"\n📝 Creating structured documents...")
        
        # 1. Per-feeder summaries (with month-specific data)
        for feeder, data in all_feeders.items():
            if not data['readings']:
                continue
            
            readings = list(data['readings'].values())
            avg = np.mean(readings)
            total = np.sum(readings)
            
            # Create overall feeder summary
            content = (
                f"Feeder: {feeder} | Location: {data['swb']} | "
                f"Average reading: {avg:.2f} KWH | "
                f"Total across all monitoring period: {total:.2f} KWH | "
                f"Readings from {min(data['readings'].keys())} to {max(data['readings'].keys())} | "
                f"Data points: {len(readings)}"
            )
            
            doc = Document(
                doc_id=str(uuid.uuid4()),
                file_id=file_id,
                file_name="JSW Energy Report",
                doc_type="feeder_data",
                content=content,
                metadata={
                    "feeder": feeder,
                    "location": data['swb'],
                    "avg_reading": float(avg),
                    "total": float(total)
                }
            )
            documents.append(doc)
        
        # 2. Daily totals
        for date, total in sorted(daily_totals.items())[:100]:  # Limit to avoid too many docs
            content = f"Date: {date} | Total plant consumption: {total:.2f} KWH"
            
            doc = Document(
                doc_id=str(uuid.uuid4()),
                file_id=file_id,
                file_name="JSW Energy Report",
                doc_type="daily_total",
                content=content,
                metadata={"date": date, "total_kwh": float(total)}
            )
            documents.append(doc)
        
        # 3. Monthly totals
        for month, total in monthly_totals.items():
            content = f"Month: {month} | Total plant consumption: {total:.2f} KWH"
            
            doc = Document(
                doc_id=str(uuid.uuid4()),
                file_id=file_id,
                file_name="JSW Energy Report",
                doc_type="monthly_total",
                content=content,
                metadata={"month": month, "total_kwh": float(total)}
            )
            documents.append(doc)
        
        # 4. Overall summary
        total_plant = sum(monthly_totals.values())
        content = (
            f"JSW MHS Electronics Plant - Complete Energy Report | "
            f"TOTAL CONSUMPTION: {total_plant:.2f} KWH | "
            f"Months covered: {len(monthly_totals)} | "
            f"Feeders monitored: {len(all_feeders)} | "
            f"Daily readings: {len(daily_totals)}"
        )
        
        doc = Document(
            doc_id=str(uuid.uuid4()),
            file_id=file_id,
            file_name="JSW Energy Report",
            doc_type="plant_summary",
            content=content,
            metadata={
                "total_consumption_kwh": float(total_plant),
                "months": len(monthly_totals),
                "feeders": len(all_feeders)
            }
        )
        documents.append(doc)
        
        print(f"\n✅ Created {len(documents)} documents:")
        print(f"   - {len(all_feeders)} feeder summaries")
        print(f"   - {len(daily_totals)} daily totals")
        print(f"   - {len(monthly_totals)} monthly totals")
        print(f"   - 1 plant summary")
        print(f"\n🔋 TOTAL PLANT CONSUMPTION: {total_plant:,.2f} KWH\n")
        
        return documents

# Test
if __name__ == "__main__":
    processor = JSWEnergyPreprocessor()
    docs = processor.process_file("Energy Consumption Daily Report MHS Ele - Copy.xlsx")
    
    print(f"\n{'='*60}")
    print("SAMPLE DOCUMENTS:")
    print(f"{'='*60}\n")
    
    for i, doc in enumerate(docs[:5], 1):
        print(f"{i}. [{doc.doc_type}]: {doc.content[:120]}...")
