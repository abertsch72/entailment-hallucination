FACTUALITY_FILE = "../../xsum_hallucination_annotations/factuality_annotations_xsum_summaries.csv"
HALLUCINATION_FILE = "../../xsum_hallucination_annotations/hallucination_annotations_xsum_summaries.csv"

import pandas

facts = pandas.read_csv(FACTUALITY_FILE)
hals = pandas.read_csv(HALLUCINATION_FILE)
facts = facts.groupby(['bbcid', 'system']).agg({'summary': 'first','is_factual': pandas.Series.mode})
print(facts)
print(facts)

import csv
summaries = {}
with open(FACTUALITY_FILE) as f:
    facts_reader = csv.reader(f, delimiter=',')
    for row in facts_reader:
        if row[0] not in summaries:
            summaries[row[0]] = {row[1]: {'summary': row[2], 'factual': row[3]}}

with open(FACTUALITY_FILE) as f:
    facts_reader = csv.reader(f, delimiter=',')
    for row in facts_reader:
        if row[0] not in summaries:
            summaries[row[0]] = {row[1]: {'summary': row[2], 'factual': row[3]}}

print(summaries)