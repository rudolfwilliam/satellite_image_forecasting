library("MODISTools")
library("tidyverse")

subset <- mt_subset(product = "VNP13A1",
                    lat = 51.322456,
                    lon = 12.823147,
                    band = "500_m_16_days_NDVI",
                    start = "2017-01-01",
                    end = "2020-01-01",
                    km_lr = 1,
                    km_ab = 1,
                    site_name = "testsite",
                    internal = TRUE,
                    progress = FALSE)

table <- matrix(, nrow = max(subset["pixel"]), ncol = nrow(unique(subset["calendar_date"])))
mt_products()
for (i in 1:max(subset["pixel"])){
  table[i,] <- t(as.vector(subset[subset["pixel"] == i,]["value"]))
}
quantiles <- sapply(data.frame(table), function(x) quantile(x, probs = c(.2,.5,.8)))
df <- data.frame(t(quantiles))

colnames(df) <- c("q1","q2","q3")

df["date"] <- as.Date(unique(subset[["calendar_date"]]),"%Y-%m-%d")


ggplot(data = df, mapping = aes(x = date)) +
  geom_line(aes(y = q2)) +
  geom_ribbon(aes(ymin = q1, ymax = q3), alpha = .3) +
  labs(y = "NDVI", x = "time") 





ndvi <- as.vector(subset["value"],mode='numeric')
dates <- as.vector(subset["calendar_date"])
