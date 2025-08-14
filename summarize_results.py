import os
import csv
from comet import download_model, load_from_checkpoint

def safe_div(num, denom):
    return num / denom if denom != 0 else 0.0

def load_comet_model(model_name="Unbabel/wmt23-cometkiwi-da-xl", gpus=1):
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model, gpus

def evaluate_file(filepath, model, gpus=1):
    # Read CSV
    records = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            records.append(row)

    # Collect COMET scores
    data = [{"src": r['Source'], "mt": r['Translation']} for r in records]
    
    comet_scores = model.predict(data, batch_size=8, gpus=gpus)
    
    # Initialize totals
    totals = {
        'term_failed': 0,
        'term_success': 0,
        'fuzzy_positive_success': 0,
        'fuzzy_positive_failed': 0,
        'fuzzy_negative_success': 0,
        'fuzzy_negative_failed': 0,
        'fuzzy_bigram_positive_success': 0,
        'fuzzy_bigram_positive_failed': 0,
    }
    composite_scores = []
    term_scores = []
    fuzzy_scores = []

    # Bucketed scores by number of terms (1-5)
    bucket_scores = {i: [] for i in range(1, 6)}
    # Bucketed scores by number of fuzzies (0-3)
    fuzzy_bucket_scores = {i: [] for i in range(0, 4)}

    print(f"Records: {len(records)}")
    for row, comet_score in zip(records, comet_scores["scores"]):
        # Parse counts
        tf = int(row['term_failed'])
        ts = int(row['term_success'])
        fps = int(row['fuzzy_positive_success'])
        fpf = int(row['fuzzy_positive_failed'])
        fns = int(row['fuzzy_negative_success'])
        fnf = int(row['fuzzy_negative_failed'])
        fbps = int(row['fuzzy_bigram_positive_success'])
        fbpf = int(row['fuzzy_bigram_positive_failed'])

        # Update totals
        totals['term_failed'] += tf
        totals['term_success'] += ts
        totals['fuzzy_positive_success'] += fps
        totals['fuzzy_positive_failed'] += fpf
        totals['fuzzy_negative_success'] += fns
        totals['fuzzy_negative_failed'] += fnf
        totals['fuzzy_bigram_positive_success'] += fbps
        totals['fuzzy_bigram_positive_failed'] += fbpf

        # Compute composite score per sentence
        part1 = safe_div(ts, ts + tf) * 5
        part2 = safe_div(fps, fps + fpf)
        part3 = safe_div(fns, fns + fnf)
        part4 = safe_div(fbps, fbps + fbpf)
        composite = (part1 + part2 + part3 + part4) / 8
        composite_scores.append(composite)

        # Term score per sentence
        term_score = safe_div(ts, ts + tf)
        term_scores.append(term_score)

        # Fuzzy score per sentence (average of parts 2-4)
        fuzzy_score = (part2 + part3 + part4) / 3
        fuzzy_scores.append(fuzzy_score)

        
        # Assign to term bucket
        term_count = ts + tf
        if 1 <= term_count <= 5:
            bucket_scores[term_count].append(term_score)

        # Assign to fuzzy bucket
        fuzzy_count = int(row.get('FuzzyCount', 0))
        if 0 <= fuzzy_count <= 3:
            fuzzy_bucket_scores[fuzzy_count].append(fuzzy_score)

    # System-wide averages
    avg_composite = sum(composite_scores) / len(composite_scores) if composite_scores else 0
    avg_term = sum(term_scores) / len(term_scores) if term_scores else 0
    avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0

    bucket_avgs = {k: (sum(v)/len(v) if v else 0) for k, v in bucket_scores.items()}
    fuzzy_bucket_avgs = {k: (sum(v)/len(v) if v else 0) for k, v in fuzzy_bucket_scores.items()}

    return {
        'comet_score': sum(comet_scores["scores"])/len(comet_scores["scores"]) if comet_scores else 0,
        'avg_composite': avg_composite,
        'avg_term': avg_term,
        'avg_fuzzy': avg_fuzzy,
        'totals': totals,
        'bucket_avgs': bucket_avgs,
        'fuzzy_bucket_avgs': fuzzy_bucket_avgs
    }

def main(input_directory, output_directory, model_name="Unbabel/wmt20-comet-qe-da"):
    model, gpus = load_comet_model(model_name)
    all_results = {}
    combined_bucket_path = os.path.join(output_directory, "all_systems_bucketed.csv")
    combined_fuzzy_bucket_path = os.path.join(output_directory, "all_systems_fuzzy_bucketed.csv")

    combined_bucket_data = {i: {} for i in range(1, 6)}
    combined_fuzzy_bucket_data = {i: {} for i in range(0, 4)}

    for fname in os.listdir(input_directory):
        if not (fname.lower().endswith(".csv") or fname.lower().endswith(".tsv")):
            continue
        path = os.path.join(input_directory, fname)

        # All rows
        result_all = evaluate_file(path, model, gpus)

        # Only rows with FuzzyCount <= 1
        filtered_records = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if int(row.get('FuzzyCount', 0)) <= 1:
                    filtered_records.append(row)
        filtered_file = path + ".filtered.tmp"
        with open(filtered_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(filtered_records)
        result_filtered = evaluate_file(filtered_file, model, gpus)

        all_results[fname] = {'all': result_all, 'fuzzy0or1': result_filtered}

        # System-wide CSV (unchanged)
        out_csv = os.path.join(output_directory, fname.replace('.csv', '_results.csv').replace('.tsv', '_results.csv'))
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'comet_score', 'avg_composite', 'avg_term', 'avg_fuzzy'] +
                            list(result_all['totals'].keys()))
            writer.writerow([model_name,
                             result_all['comet_score'],
                             result_all['avg_composite'],
                             result_all['avg_term'],
                             result_all['avg_fuzzy']] +
                            list(result_all['totals'].values()))

        # Store bucketed term results
        for bucket_num in range(1, 6):
            combined_bucket_data[bucket_num][fname] = result_all['bucket_avgs'].get(bucket_num, 0)
        # Store fuzzy bucket results
        for bucket_num in range(0, 4):
            combined_fuzzy_bucket_data[bucket_num][fname] = result_all['fuzzy_bucket_avgs'].get(bucket_num, 0)

        print(f"File: {fname}")
        print("All sentences:", result_all)
        print("FuzzyCount <=1 sentences:", result_filtered)

    # Write combined term bucketed results
    all_systems = sorted(all_results.keys())
    with open(combined_bucket_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['terms'] + all_systems)
        for bucket_num in range(1, 6):
            row = [bucket_num] + [combined_bucket_data[bucket_num].get(sys, 0) for sys in all_systems]
            writer.writerow(row)

    # Write combined fuzzy bucketed results
    with open(combined_fuzzy_bucket_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fuzzies'] + all_systems)
        for bucket_num in range(0, 4):
            row = [bucket_num] + [combined_fuzzy_bucket_data[bucket_num].get(sys, 0) for sys in all_systems]
            writer.writerow(row)

    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize COMET QE scores for CSV reports")
    parser.add_argument("--input_directory", help="Directory containing report files")
    parser.add_argument("--output_directory", help="Directory to save analysis to")
    parser.add_argument("--model", default="Unbabel/wmt20-comet-qe-da", help="COMET reference-free model name")
    args = parser.parse_args()
    main(args.input_directory, args.output_directory,model_name=args.model)
