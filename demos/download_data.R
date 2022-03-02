subset <- mt_subset(product = "VNP13A1",
                    lat = 51.322456,
                    lon = 12.823147,
                    band = "500_m_16_days_NDVI",
                    start = "2017-01-01",
                    end = "2019-01-01",
                    km_lr = 1,
                    km_ab = 1,
                    site_name = "testsite",
                    internal = TRUE,
                    progress = FALSE)
pix_1_data <-subset[subset["pixel"]==4,]
ggplot(pix_1_data, aes(x = calendar_date,y=value)) +  geom_point()




ndvi <- as.vector(subset["value"],mode='numeric')
dates <- as.vector(subset["calendar_date"])



for date in dates{
  
}
pixels <- as.vector(subset["pixel"])

ggplot2(ndvi)
