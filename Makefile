# Remove target files after command failure.
.DELETE_ON_ERROR:

models: \
		enwiki_models

tuning_reports: \
		enwiki_tuning_reports

touch:
	touch datasets/*
	touch models/*


############################# English Wikipedia ################################

# Pull down data from wikilabels
datasets/enwiki.human_labeled_sessions.2k_2018.json:
	./utility fetch_labels \
		https://labels.wmflabs.org/campaigns/enwiki/86/ \
		 --verbose > $@

# Extracting into features
datasets/enwiki.human_labeled_sessions.features.2k_2018.json: \
		datasets/enwiki.human_labeled_sessions.2k_2018.json

	./utility newcomer_extract \
		--dump-file $< \
		--host https://en.wikipedia.org \
		--verbose > $@

# Create scaling mapper
models/enwiki.goodfaith.scaling.mapper: \
		datasets/enwiki.human_labeled_sessions.features.2k_2018.json
	./utility newcomer_train \
		--dump-file $< \
		--fn make_scaling_mapper \
	    --scaling_mapper $@

#  Reports
tuning_reports/enwiki.goodfaith.md: \
		models/enwiki.goodfaith.scaling.mapper
	./utility newcomer_train \
		--dump-file datasets/enwiki.human_labeled_sessions.features.2k_2018.json \
		--fn tuning_report \
		--model_params config/classifiers.params.json \
		--scaling_mapper models/enwiki.goodfaith.scaling.mapper \
		> $@

# Create model
models/enwiki.goodfaith.logistic_regression.model: \
		tuning_reports/enwiki.goodfaith.md
	./utility newcomer_train \
		--dump-file datasets/enwiki.human_labeled_sessions.features.2k_2018.json \
		--fn create_model \
		--model_params config/model_defaults.json \
		--scaling_mapper models/enwiki.goodfaith.scaling.mapper \
	    --model_file $@

enwiki_models: \
	models/enwiki.goodfaith.logistic_regression.model

enwiki_tuning_reports: \
	tuning_reports/enwiki.goodfaith.md
