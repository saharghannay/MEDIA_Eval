#
# This file is part of Data.

Data is a directory containing all data developped at LIMSI for SLU purposes on the MEDIA corpus. 
It containes the MEDIA data with the different embeddings (Camembert, cbow (trained on different data)) used for the experiments (for more detailes see the paper).
MEDIA CORPUS is the property of ELRA and has to be acquired before using tha data. See : https://catalog.elra.info/en-us/repository/browse/ELRA-S0272/ (MEDIA is publicly available only for academic use).


##Data preparation 

Before starting the experiments, please follow the instruction to prepare the data. 
The full prepared data can be downloaded from [here](https://perso.limsi.fr/ghannay/Data.zip) 

## Run the following scripts to prepare the data used for the configuration "freeze embeddings"
- Embeddings trained on media data
	`scripts/prepare_MEDIA_data.csh `
- Embeddings trained on wiki data
	`scripts/prepare_MEDIA_data_wiki.csh `
- Embeddings trained on wiki + media data
	`scripts/prepare_MEDIA_data_wiki.csh `
- Cammebert Embeddings 
	1- Get Camembert Embeddings 
		`scripts/get_Camembert_embed_media.csh`
	2- Prepare the data 
		`scripts/prepare_MEDIA_data_camembert_embed.csh`