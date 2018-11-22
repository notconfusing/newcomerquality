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
		--verbose # > $@

# Tuning Reports
tuning_reports/enwiki.damaging.md: \
		datasets/enwiki.human_labeled_sessions.features.2k_2018.json
	cat $< | \
	./utility newcomer_tune \
		config/classifiers.params.yaml \
		editquality.feature_lists.enwiki.damaging \
		damaging \
		roc_auc.labels.true \
		--label-weight "true=$(damaging_weight)" \
		--pop-rate "true=0.034163555464634586" \
		--pop-rate "false=0.9658364445353654" \
		--center --scale \
		--cv-timeout 60 \
		--debug > $@

# Create model
models/enwiki.goodfaith.logistic_regression.model: \
		datasets/enwiki.human_labeled_sessions.features.2k_2018.json
	cat $^ | \
	./utility newcomer_train \
		revscoring.scoring.models.GradientBoosting \
		editquality.feature_lists.enwiki.damaging \
		damaging \
		--version=$(damaging_major_minor).0 \
		-p 'learning_rate=0.01' \
		-p 'max_depth=7' \
		-p 'max_features="log2"' \
		-p 'n_estimators=700' \
		--label-weight "true=$(damaging_weight)" \
		--pop-rate "true=0.034163555464634586" \
		--pop-rate "false=0.9658364445353654" \
		--center --scale > $@

	./utility model_info $@ > model_info/enwiki.goodfaith.md

enwiki_models: \
	models/enwiki.goodfaith.logistic_regression.model

enwiki_tuning_reports: \
	tuning_reports/enwiki.goodfaith.md
