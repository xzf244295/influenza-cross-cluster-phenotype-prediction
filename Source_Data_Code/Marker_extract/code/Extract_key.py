

#####按照关键词进行提取。

import pandas as pd
import re

df = pd.read_csv("../data/abs_pubmed_influenza_abstracts.csv")

aa3 = [
"Ala","Arg","Asn","Asp","Cys","Glu","Gln","Gly","His","Ile",
"Leu","Lys","Met","Phe","Pro","Ser","Thr","Trp","Tyr","Val"
]

aa3_pattern = "|".join(aa3)

pattern1 = rf'\b(?:[A-Z]\d+[A-Z]|(?:{aa3_pattern})\d+(?:{aa3_pattern}))\b'
pattern2 = rf'\b(?:[A-Z]|(?:{aa3_pattern}))\s*to\s*(?:[A-Z]|(?:{aa3_pattern}))\b'
pattern3 = rf'\b(?:[A-Z]\d*|(?:{aa3_pattern})\d*)\s*--?>\s*(?:[A-Z]\d*|(?:{aa3_pattern})\d*)\b'
pattern4 = rf'\b(?:[A-Z]|(?:{aa3_pattern}))\s*-\s*(?:[A-Z]|(?:{aa3_pattern}))\b'
pattern5 = rf'\b(?:{aa3_pattern}|[A-Z])(?:\(\d+\)|\d+)?\s*-?to\s*-?(?:{aa3_pattern}|[A-Z])(?:\(\d+\)|\d+)?\b'
# pattern6 = r'\b(mutation|mutations|substitution|mutant|variant|variants)\b'
pattern6 = rf'(?<![A-Z])\b\d+(?:[ACDEFGHIKLMNPQRSTVWY]|(?:{aa3_pattern}))\b'

patterns = [pattern1,pattern2,pattern3,pattern4,pattern5,pattern6]

results = []

for i,row in df.iterrows():

    text = str(row["Title"]) + " " + str(row["Abstract"])

    markers = []

    for p in patterns:
        matches = re.findall(p,text,flags=re.IGNORECASE)
        markers.extend(matches)

    if markers:

        results.append({
            "PMID":row["PMID"],
            "Title":row["Title"],
            "Abstract":row["Abstract"],
            "Markers":",".join(set(markers))
        })

out = pd.DataFrame(results)


out.to_csv("key_markers.csv",index=False)

print("Finished")
print("Articles with markers:",len(out))