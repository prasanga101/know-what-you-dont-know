from datasets import load_dataset

belebele_ne = load_dataset("facebook/belebele", "npi_Deva")
belebele_bn = load_dataset("facebook/belebele", "ben_Beng")
belebele_en = load_dataset("facebook/belebele", "eng_Latn")
squad_v2 = load_dataset("rajpurkar/squad_v2")
