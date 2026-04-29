# COLESLAW
https://www.clarin.si/repository/xmlui/handle/11356/2095#

curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2095{/COLESLAW.zip}

```json
{ // primer
  "id": 128390,
  "naziv": "Zakon o osebnem imenu (ZOI-1)",
  "mopedId": "ZAKO3890",
  "eva": "2004-1711-0017",
  "epa": "0486-IV",
  "sop": "2006-01-0746",
  "text": "Opozorilo: Besedilo osnovnega predpisa\nZAKON\nO OSEBNEM IMENU (ZOI-1)\nI. SPLOŠNE DOLOČBE\n1. člen\n(vsebina zakona)\nTa zakon določa pojem, sestavo in določitev osebnega imena ter pogoje za njegovo uporabo in spremembo za državljanke in državljane Republike Slovenije (v nadaljnjem besedilu: državljan).\n2. člen\n(pojem osebnega imena)\n(1) Osebno ime je pravica državljana in služi za razločevanje ter identifikacijo fizičnih oseb. Državljan je osebno ime dolžan uporabljati.\n(2) Osebno ime državljanu zagotavlja identiteto, varstvo njegove osebnosti in dostojanstva.\n........"
}
```

# PISRS
https://pisrs.si/

has api for downloading stuff, needs application for access: https://pisrs.si/swagger

# run

pisrs_tematsko_kazalo_delovno_pravo.csv - documents from pisrs tematsko kazalo under delovno pravo

pisrs.py - collect all documents into one pisrs.csv and embed them into doc_embs.npy
query_similarity.py - embed queries and save similarity for each document for each query in similarities.csv
filter_pisrs.py - use similarities.csv to filter pisrs.csv into filtered_pisrs.csv
build_index.py - use filtered_pisrs.csv to chunk the documents and create FAISS index 
run_rag.py - run simple rag test

