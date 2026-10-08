"""
Data Loader and Integration Module for Albania Public Procurement Commission 2025 Dataset.
Merges main tenders with parties, CPV item codes, fees, and complaint identifiers.
"""

import os
import csv
from collections import defaultdict


def load_raw_data(data_dir: str):
    """
    Load raw CSV files using standard library csv.DictReader.
    """
    files = {
        'main': 'main.csv',
        'parties': 'parties.csv',
        'items': 'tender_items.csv',
        'fees': 'tender_participationFees.csv',
        'complaints': 'complaints_identifiers.csv'
    }

    data = {}
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            reader = list(csv.DictReader(f))
            data[key] = reader

    return data


def integrate_datasets(data_dir: str):
    """
    Integrates and merges main.csv with parties, items, fees, and complaints.
    Returns a list of dicts representing unified procurement records.
    """
    raw_data = load_raw_data(data_dir)

    # 1. Aggregate parties by _link_main
    parties_by_link = defaultdict(list)
    for party in raw_data['parties']:
        link_main = party.get('_link_main') or party.get('_link')
        if link_main:
            parties_by_link[link_main].append(party)

    # 2. Aggregate CPV codes by _link_main
    items_by_link = defaultdict(list)
    for item in raw_data['items']:
        link_main = item.get('_link_main') or item.get('_link')
        cpv = item.get('cpv_code', '').strip()
        if link_main and cpv:
            items_by_link[link_main].append(cpv)

    # 3. Aggregate participation fees by _link_main
    fees_by_link = defaultdict(list)
    for fee in raw_data['fees']:
        link_main = fee.get('_link_main') or fee.get('_link')
        if link_main:
            fees_by_link[link_main].append(fee)

    # 4. Integrate into unified records
    unified_records = []
    for main_rec in raw_data['main']:
        rec = dict(main_rec)
        link = rec.get('_link', '')

        # Parties aggregation
        rel_parties = parties_by_link.get(link, [])
        party_names = [p.get('name', '') for p in rel_parties if p.get('name')]
        party_roles = [p.get('roles', '') for p in rel_parties if p.get('roles')]
        rec['integrated_party_names'] = '; '.join(filter(None, party_names))
        rec['integrated_party_roles'] = '; '.join(filter(None, party_roles))
        rec['party_count'] = len(rel_parties)

        # CPV items aggregation
        cpv_list = items_by_link.get(link, [])
        rec['cpv_codes'] = '; '.join(cpv_list)
        rec['cpv_count'] = len(cpv_list)

        # Fee aggregation
        rel_fees = fees_by_link.get(link, [])
        fee_amounts = [f.get('value_amount', '0') for f in rel_fees if f.get('value_amount')]
        fee_val_str = fee_amounts[0] if fee_amounts else '0'
        rec['participation_fee_amount'] = fee_val_str
        try:
            rec['has_participation_fee'] = 1 if float(fee_val_str or 0) > 0 else 0
        except ValueError:
            rec['has_participation_fee'] = 0

        # Numeric conversions
        try:
            rec['tender_value_amount_num'] = float(rec.get('tender_value_amount') or 0.0)
        except ValueError:
            rec['tender_value_amount_num'] = 0.0

        unified_records.append(rec)

    return unified_records


if __name__ == '__main__':
    data_path = r'c:\Users\varsh\Downloads\albania_public_procurement_commission_2025.csv\2025'
    integrated = integrate_datasets(data_path)
    print(f"Successfully integrated {len(integrated)} records.")
    print("Sample record keys:", list(integrated[0].keys()))
