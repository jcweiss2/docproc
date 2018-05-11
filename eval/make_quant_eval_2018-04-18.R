library(cluster)
library(clusterCrit)
library(dplyr)
library(proxy)

if (!("clusters_tb" %in% ls())) {
  cat("Loading clusters_tb\n")
  clusters_tb = read.csv("../data/180415clusters.csv", stringsAsFactors=FALSE) %>% tbl_df
  clusters_tb = clusters_tb %>% select(-X) %>% mutate(cluster = as.integer(cluster))
}

if (!("providers_tb" %in% ls())) {
  cat("Loading providers_tb\n")
  providers_tb = readRDS("../data/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")
}

if (!("hidden_tb" %in% ls())) {
  cat("Loading hidden_tb\n")
  # hidden_tb = read.csv("../data/180416npihidden.csv", stringsAsFactors=FALSE) %>% tbl_df
  # saveRDS(hidden_tb, file="../data/180416npihidden.rds")
  hidden_tb = readRDS("data/180416npihidden.rds")

  hidden_x = as.matrix(hidden_tb[,sprintf("X%d", 0:127)])
  rownames(hidden_x) = hidden_tb$npi
}

specialties_tb = providers_tb %>% select(npi, specialty_description) %>%
                                  mutate(specialty_id = as.integer(specialty_description))


both_tb = clusters_tb %>% left_join(specialties_tb, by="npi")

# Compute silhoutte plots.
cossim = function(x, y) {
  x %*% t(y) / sqrt(x%*%t(x) * y%*%t(y))
}
#hidden_dist = cossim(hidden_x, hidden_x)

# Subsample 1% of the providers.
sub_npi = unlist(sapply(unique(clusters_tb$cluster), function(c) {
  rn = (clusters_tb %>% filter(cluster == c))$npi
  sub_rn = sample(rn, round(length(rn)*0.01))
  sub_rn
}))
sub_both_tb = both_tb %>% filter(npi %in% sub_npi)
sub_hidden_x = hidden_x[as.character(sub_npi),]

# Compute cossine similarity.
sub_cossim = dist(sub_hidden_x, sub_hidden_x, method="cosine")

# Make silhoutte plot.
cluster_sil = silhouette(sub_both_tb$cluster, sub_cossim)
speciality_sil = silhouette(sub_both_tb$specialty_id, sub_cossim)
pdf("silhouette.pdf")
plot(cluster_sil, main="Silhouette Plot for Our Clustering")
plot(speciality_sil, main="Silhouette Plot based on Specialties")
dev.off()

# 
sub_ext = extCriteria(sub_both_tb$cluster, sub_both_tb$specialty_id, "all")
write.csv(int_tb, file="external_criteria_on_subset.csv", row.names=FALSE)

cluster_int = intCriteria(sub_hidden_x, sub_both_tb$cluster, "all")
specialty_int = intCriteria(sub_hidden_x, sub_both_tb$specialty_id, "all")

cluster_int_tb = data_frame(key=names(cluster_int), value_for_cluster=unlist(cluster_int))
specialty_int_tb = data_frame(key=names(specialty_int), value_for_specialty=unlist(specialty_int))
int_tb = cluster_int_tb %>% outer_join(specialty_int_tb, by="key")
write.csv(int_tb, file="internal_criteria_on_subset.csv", row.names=FALSE)

alskdjfalskdjf

tb = read.csv("../180415clusters.csv") %>% tbl_df
old_tb = tb
tb = tb %>% select(-X) %>% mutate(cluster = as.integer(cluster))

if (0) {
  # puf = read.csv("~/Downloads/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.csv", header=TRUE) %>% tbl_df
  # saveRDS(puf, "~/Downloads/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")
  puf = readRDS("~/Downloads/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")

  medicare_clusters = puf %>% select(npi, specialty_description)
  write.csv(medicare_clusters, file="medicare_clusters.csv", row.names=FALSE)
} else {
  medicare_clusters = read.csv("medicare_clusters.csv")
}
medicare_clusters = medicare_clusters %>% mutate(specialty_id = as.integer(specialty_description))

both = tb %>% left_join(medicare_clusters, by="npi")
write.csv(both, file="both.csv", row.names=FALSE)

hidden = read.csv("~/Downloads/180416npihidden.csv") %>% tbl_df

(ext = extCriteria(both$cluster, both$specialty_id, "Rand"))

med_counts = both %>% group_by(specialty_description) %>% summarise(num_spec = n()) %>% arrange(-num_spec)
our_counts = both %>% group_by(cluster) %>% summarise(num_clus = n()) %>% arrange(-num_clus)
both_counts = both %>% group_by(cluster, specialty_description) %>% summarise(num_both = n()) %>% arrange(-num_both)
counts = both_counts %>% left_join(med_counts, by="specialty_description") %>%
                         left_join(our_counts, by="cluster")
counts = counts %>% mutate(frac_both_in_spec=num_both/num_spec, frac_both_in_clus=num_both/num_clus)
write.csv(counts, file="180415clusters_eval_counts.csv", row.names=FALSE)


#extCriteria(both$cluster, both$specialty_id, c("Jaccard"))

# colwidths = c(10, 70, 20, 1, 20, 1, 1, 55, 55, 40, 5, 4, 2, 2, 75, 1, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 1, 8, 1, 8, 8, 1, 8, 8, 1, 8, 8, 1, 8, 8, 1, 8, 8, 1, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 1, 8, 1, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8)

# colnames = c("npi", "nppes_provider_last_org_name", "nppes_provider_first_name", "nppes_provider_mi", "nppes_credentials", "nppes_provider_gender", "nppes_entity_code", "nppes_provider_street1", "nppes_provider_street2", "nppes_provider_city", "nppes_provider_zip5", "nppes_provider_zip4", "nppes_provider_state", "nppes_provider_country", "specialty_description", "description_flag", "medicare_prvdr_enroll_status", "total_claim_count", "total_30_day_fill_count", "total_drug_cost", "total_day_supply", "bene_count", "ge65_suppress_flag", "total_claim_count_ge65", "total_30_day_fill_count_ge65", "total_drug_cost_ge65", "total_day_supply_ge65", "bene_count_ge65_suppress_flag", "bene_count_ge65", "brand_suppress_flag", "brand_claim_count", "brand_drug_cost", "generic_suppress_flag", "generic_claim_count", "generic_drug_cost", "other_suppress_flag", "other_claim_count", "other_drug_cost	", "mapd_suppress_flag", "mapd_claim_count", "mapd_drug_cost", "pdp_suppress_flag", "pdp_claim_count", "pdp_drug_cost", "lis_suppress_flag", "lis_claim_count", "lis_drug_cost", "nonlis_suppress_flag", "nonlis_claim_count", "nonlis_drug_cost", "opioid_claim_count", "opioid_drug_cost", "opioid_day_supply", "opioid_bene_count", "opioid_prescriber_rate", "antibiotic_claim_count", "antibiotic_drug_cost", "antibiotic_bene_count", "hrm_ge65_suppress_flag", "hrm_claim_count_ge65", "hrm_drug_cost_ge65", "hrm_bene_ge65_suppress_flag", "hrm_bene_count_ge65", "antipsych_ge65_suppress_flag", "antipsych_claim_count_ge65", "antipsych_drug_cost_ge65", "antipsych_bene_ge65_suppress_flg", "antipsych_bene_count_ge65", "average_age_of_beneficiaries", "beneficiary_age_less_65_count", "beneficiary_age_65_74_count", "beneficiary_age_75_84_count", "beneficiary_age_greater_84_count", "beneficiary_female_count", "beneficiary_male_count", "beneficiary_race_white_count", "beneficiary_race_black_count", "beneficiary_race_asian_pi_count", "beneficiary_race_hispanic_count", "beneficiary_race_nat_ind_count", "beneficiary_race_other_count", "beneficiary_nondual_count", "beneficiary_dual_count", "beneficiary_average_risk_score")

# puf = read.fwf("~/Downloads/PartD_Prescriber_PUF_NPI_15/PartD_Prescriber_PUF_NPI_15.txt", widths=colwidths, header=TRUE)


