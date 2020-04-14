library('socialmixr')
data(polymod)

age_bin_edges <- c($age_bin_edges)
age_distribution <- c($age_distribution)
age_dist <- data.frame(age_bin_edges, age_distribution)

names(age_dist)[names(age_dist) == 'age_bin_edges'] <- 'lower.age.limit'
names(age_dist)[names(age_dist) == 'age_distribution'] <- 'population'

age_limits = c()
for(v in age_dist['lower.age.limit']) { age_limits <- v }

m <- contact_matrix(polymod,
                    countries = 'United Kingdom',
                    age.limits = age_limits,
                    n=10,
                    survey.pop=age_dist)

mr <- Reduce("+", lapply(m$matrices, function(x) {x$matrix})) / length(m$matrices)
mr <- as.data.frame(mr)
