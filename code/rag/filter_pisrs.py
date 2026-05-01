import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_colwidth", 110)

df = pd.read_csv("similarities.csv")
_pisrs = pd.read_csv("pisrs.csv")
pisrs = _pisrs.drop(columns=['id', 'mopedId', 'eva', 'epa', 'text'])

sim_cols = [col for col in df.columns if col.startswith("sim_")]
X = df[sim_cols].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df["pc1"] = X_pca[:, 0]
df["pc2"] = X_pca[:, 1]

df["delovno_pravo"] = df["sop"].map(
    pisrs.set_index("sop")["delovno_pravo"]
)
df["delovno_pravo"] = df["delovno_pravo"].fillna(False)

TOP_K = 750

total_positive = df["delovno_pravo"].sum()
idxs = [
    0,  # pogodba o zaposlitvi sklenitev pogoji delovno razmerje
    # # 1,  # vrste pogodb o zaposlitvi določila pravice obveznosti
    # 2,  # odpoved pogodbe o zaposlitvi odpovedni rok prenehanje delovnega razmerja
    3,  # izredna odpoved delavec delodajalec razlogi za odpoved
    # 4,  # plača nadomestilo regres izplačilo plače minimalna plača
    # # 5,  # zamuda pri izplačilu plače pravice delavca
    # 6,  # delovni čas nadure razpored dela počitek tedenski in dnevni
    7,  # organizacija delovnega časa nočno delo zakon omejitve
    8,  # dopust letni dopust bolniška odsotnost z dela pravice delavca
    # 9,  # pravice delavcev varstvo delavca diskriminacija mobing delovno mesto
    10, # varnost in zdravje pri delu obveznosti delodajalca zaščita delavca
    11, # zaposlovanje tujcev delovna dovoljenja pogoji za delo tujcev
    # # 12, # delo tujcev v Sloveniji omejitve in postopki
    # 13, # obveznosti delodajalca pogodba o zaposlitvi varnost delovno mesto
    # 14, # odgovornosti delodajalca sankcije kršitve delovnega prava
    15, # kolektivna pogodba delovno pravo sindikati
    # # 16, # kolektivne pogodbe plače tarifni del
    # 17, # aneks kolektivna pogodba spremembe
    # # 18, # javni sektor plače kolektivne pogodbe RTV zdravstvo
    # 19, # pravilnik uredba zaposlovanje delovna razmerja
    20, # uskladitev plač bruto minimalna mesečna plača odredba delavci
    # 21, # bruto plača neto plača izračun plače davki prispevki
]
print(idxs)

df["sim"] = df[[sim_cols[i] for i in idxs]].max(axis=1)
pisrs["sim"] = pisrs["sop"].map(
    df.set_index("sop")["sim"]
)
pisrs["sim"] = pisrs["sim"].fillna(0)

top_dfs = []
rest_dfs = []
topk_sets = {}
for ci, col in enumerate(sim_cols):
    sorted_df = df.sort_values(col, ascending=False)

    top_k_df = sorted_df.head(TOP_K).copy()
    rest_df = sorted_df.iloc[TOP_K:].copy()

    top_dfs.append(top_k_df)
    rest_dfs.append(rest_df)

    topk_sets[ci] = set(top_k_df["sop"])
    # print(sorted_df)
    # print(top_k_df)

top_df = pd.concat([t for ti, t in enumerate(top_dfs) if ti in idxs]).drop_duplicates(subset="sop")
rest_df = pd.concat([r for ri, r in enumerate(rest_dfs) if ri in idxs]).drop_duplicates(subset="sop")
top_ids = set(top_df["sop"])
rest_df = rest_df[~rest_df["sop"].isin(top_ids)]

for ci, col in enumerate(sim_cols):
    dp_count = top_dfs[ci]["delovno_pravo"].sum()
    dp_recall = dp_count / total_positive if total_positive > 0 else 0
    current = topk_sets[ci]
    others = set().union(*[s for c, s in topk_sets.items() if c != ci and c in idxs])
    exclusive = current - others
    exclusive_dp = df[df["sop"].isin(exclusive) & df["delovno_pravo"]]

    print(f"{col}:")
    print(f"                     used: {ci in idxs}")
    print(f"                 dp count: {dp_count}")
    print(f"                dp recall: {dp_recall:.4f}")
    print(f"           exclusive docs: {len(exclusive)}")
    print(f"  exclusive delovno_pravo: {len(exclusive_dp)}")
    print(f"       exclusive dp ratio: {len(exclusive)} / {len(exclusive_dp)} = {(len(exclusive_dp)/len(exclusive)):.3f}")
    print(f"entries from delovno pravo this would add:")
    if ci not in idxs:
        if len(exclusive_dp) > 0:
            print(exclusive_dp)
        else:
            print("Empty DataFrame")
    print()

print()
# print(top_df)
# print(rest_df)
print()
print("selected:", top_df["delovno_pravo"].sum())
print("unselected:", rest_df["delovno_pravo"].sum())
print()

unmatched_delovno = rest_df[rest_df["delovno_pravo"]]
unmatched_undelovno = rest_df[~rest_df["delovno_pravo"]]
matched_delovno = top_df[top_df["delovno_pravo"]]
matched_undelovno = top_df[~top_df["delovno_pravo"]]

print("unmatched_delovno\n", unmatched_delovno)
# print("unmatched_undelovno\n", unmatched_undelovno)
# print("matched_delovno\n", matched_delovno)
# print("matched_undelovno\n", matched_undelovno)

plt.scatter(rest_df["pc1"], rest_df["pc2"], c="gray", alpha=0.5, s=10)
plt.scatter(top_df["pc1"], top_df["pc2"], c=top_df["sim"], alpha=0.5, s=10)

# plt.scatter(df["pc1"], df["pc2"], c=df["sim"], alpha=0.5, s=10)

plt.scatter(df[df["delovno_pravo"]]["pc1"], df[df["delovno_pravo"]]["pc2"], marker=".", color="red", s=10, label="Delovno pravo documents")

plt.colorbar(label="Maximum similarity")
plt.legend()
plt.savefig("pisrs_pca.png", dpi=300, bbox_inches="tight")
# plt.show()

filtered_pisrs = _pisrs[_pisrs["sop"].isin(top_df["sop"]) | (_pisrs["delovno_pravo"] == True)]
filtered_pisrs.to_csv("filtered_pisrs.csv", index=False)
print(filtered_pisrs)

