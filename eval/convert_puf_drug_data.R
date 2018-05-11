library(dplyr)
library(reshape2)

# Load output data.
if (!("output_tb" %in% ls())) {
  cat("Loading output_tb\n")
  header_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/p6iedds3xiaht1nmhlval5tzq0k9dqcx.gz | head -n 2")) %>% tbl_df
  output_col_classes = c("integer", rep("numeric", ncol(header_tb)-1))
  output_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/p6iedds3xiaht1nmhlval5tzq0k9dqcx.gz"), colClasses=output_col_classes) %>% tbl_df
  output_x = as.matrix(output_tb %>% select(-npi))
  rownames(output_x) = output_tb$npi
}

# Make non-zero count data and log data.
full_tb = melt(output_tb, "npi")%>% tbl_df
count_tb = full_tb %>% filter(value != 0) %>% rename(count = value) %>% mutate(log_count = log(count))

# Load clusters.
if (!("clusters_tb" %in% ls())) {
  cat("Loading clusters_tb\n")
  clusters_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/180422clusters40merged.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  clusters_tb = clusters_tb %>% select(-X) %>% mutate(cluster = as.integer(cluster))
}

# Load kmeans clusters.
if (!("kmeans_tb" %in% ls())) {
  cat("Loading kmeans_tb\n")
  kmeans_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/kmeans_2018-04-20.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  kmeans_tb = kmeans_tb %>% select(-X) %>% mutate(kmeans_cluster = as.integer(cluster)) %>% select(-cluster)
}

# Load providers data.
if (!("providers_tb" %in% ls())) {
  cat("Loading providers_tb\n")
  providers_tb = readRDS("../data/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")
}

# Make specialties_tb.
cat("Make specialties_tb.\n")
specialties_tb = providers_tb %>% select(npi, specialty_description) %>%
                                  mutate(specialty_id = as.integer(specialty_description))

# Summarize counts by clusters.
joined_tb = clusters_tb %>% left_join(count_tb, by="npi")
cluster_to_drugs_tb = joined_tb %>% group_by(cluster, variable) %>%
                                    summarise(num_providers = n(), sum_counts = sum(count), sum_log_counts = sum(log_count))
write.csv(cluster_to_drugs_tb, file="../data/180422clusters40merged_to_drug_counts.csv", row.names=FALSE)

# Summarize counts by kmeans clusters.
joined_tb = kmeans_tb %>% left_join(count_tb, by="npi")
kmeans_to_drugs_tb = joined_tb %>% group_by(kmeans_cluster, variable) %>%
                                   summarise(num_providers = n(), sum_counts = sum(count), sum_log_counts = sum(log_count))
write.csv(kmeans_to_drugs_tb, file="../data/kmeans_2018-04-20_to_drug_counts.csv", row.names=FALSE)

# Summarize counts by specialties.
joined_tb = specialties_tb %>% left_join(count_tb, by="npi")
specialties_to_drugs_tb = joined_tb %>% group_by(specialty_description, specialty_id, variable) %>%
                                summarise(num_providers = n(), sum_counts = sum(count), sum_log_counts = sum(log_count))
write.csv(specialties_to_drugs_tb, file="../data/specialties_to_drug_counts.csv", row.names=FALSE)


