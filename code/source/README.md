# COLESLAW 1.0
[COLESLAW 1.0](https://www.clarin.si/repository/xmlui/handle/11356/2095#) is a large-scale collection of Slovenian legal texts compiled from authoritative public sources. The corpus covers legislative, judicial, and governmental legal documents and is designed to support research in legal NLP, information retrieval, contradiction detection, legal reasoning, and domain adaptation of language models.

Download and unzip after:
```sh
curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2095{/COLESLAW.zip}
```

---

Example of a document entry:
```json
{ 
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
Our primary source is [PISRS (Pravni informacijski sistem Republike Slovenije - Legal Information System of the Republic of Slovenia)](https://pisrs.si/), which contains all legal documents relating to slovenian legislation.

 <!-- API for live data: https://pisrs.si/swagger -->

