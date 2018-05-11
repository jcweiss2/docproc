library(dplyr)

# Load clusters.
if (!("clusters_tb" %in% ls())) {
  cat("Loading clusters_tb\n")
  clusters_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/180426clusters60merged.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  clusters_tb = clusters_tb %>% select(-X) %>% mutate(cluster = as.integer(cluster))
}

# Load NPI-to-HCPCS table.
if (!("npi_to_hcpcs_tb" %in% ls())) {
  cat("Loading npi_to_hcpcs_tb\n")
  npi_to_hcpcs_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/npi_to_hcpcs_code.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  #npi_to_hcpcs_tb = npi_to_hcpcs_tb %>% mutate(cluster = as.integer(cluster))
  npi_to_hcpcs_tb = npi_to_hcpcs_tb %>% rename(npi = National.Provider.Identifier)
}

# Make counts of HCPCS codes in each cluster.
merged_tb = clusters_tb %>% left_join(npi_to_hcpcs_tb, by="npi")
counts_tb = merged_tb %>% group_by(cluster, HCPCS.Code, HCPCS.Description) %>%
                          summarise(num_providers = n(),
                                    sum_counts = sum(Number.of.Services),
                                    sum_log_counts = sum(log(Number.of.Services)))
write.csv(counts_tb, file="../data/180426clusters60merged_to_hcpcs_counts.csv", row.names=FALSE)
