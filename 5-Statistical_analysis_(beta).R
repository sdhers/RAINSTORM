# Graphic and statistic analysis of Object Recognition Tasks in mice.

# Introduction -----------------------------------------------------------------

#### Setup ####

ls()
rm(list=ls())
ls()

# libraries

wd <- getwd()
example_folder <- file.path(wd, "examples", "PD")
setwd(example_folder)

list.of.packages <- c(
  "ggplot2", "reshape2", "corrplot", "nlme", "emmeans", "multcomp", "car", "dplyr", "doBy", "Rmisc", "ggeffects", "sjPlot", "lme4", "glmmTMB", "DHARMa", "pastecs", "psych", "effects", "margins", "GGally", "gridExtra", "MuMIn", "performance",  "lmerTest", "Hmisc", "ggsignif", "readxl", "readr")

# install.packages("easypackages")
easypackages::packages(list.of.packages)


#### Load the data ####

df <- read_csv("results.csv")
ID <- df %>% select(Video, Group, Trial, DI_final)
TS <- ID %>% filter(Trial == "TS")

class(TS)     # bject type
str(TS)       # structure

TS$Video <- factor(TS$Video)
TS$Group <- factor(TS$Group, levels = c("PD", "Veh"))
TS$Trial <- factor(TS$Trial, levels = c("Hab", "TR", "TS"))
TS$DI_final <- as.numeric(TS$DI_final)

str(TS)
dim(TS)       # Dimensions
head(TS)      # First rows
summary(TS)   # Variable summary

outliers <- c("")  # Add all the values you want to remove
TS <- TS[!TS$Video %in% outliers, ]

TS$DI_scaled <- TS$DI_final / 100 # Divide percentages by 100 to get numbers between 0 and 1 (for beta distribution)

# Plot

ggplot(TS, aes(x = Group, y = DI_final)) + 
  geom_boxplot(outlier.size = 2, outlier.shape = 4, width = 0.3) + # Outliers
  stat_summary(fun = "mean", geom = "point", shape = 16, size = 2) +
  geom_jitter(data = TS, width = 0.1, size = 1.5, aes(color = Group, shape = Group)) +
  geom_hline(aes(yintercept=50), colour="grey", size=1.5, alpha = 0.8) + # Línea en 50
  ggtitle("Discrimination Index") + # Título
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("DI (%)") + theme(axis.title.y = element_text(angle = 90))


#### Function to evaluate assumptions ####

assumptions <- function(model) {
  
  res <- resid(model, type = "pearson")
  pred <- predict(model)
  
  par(mfrow = c(1, 2))
  
  plot(x = pred,
       y = res,
       # ylim = c(-4, 4),
       xlab = "Predicted",
       ylab = "Pearson Residuals",
       main = "res v. pred dispersion", 
       cex.main = 0.8 )
  
  abline(h = c(-2, 2, 0),
         col = c("red", "red", "black"),
         lty = c(2, 2, 1))
  
  #  qqnorm(res, cex.main = 0.8)
  #  qqline(res)
  qqPlot(res)
  
  return(list(
    shapiro.test(res),
    leveneTest(res, pred)))
}

##### Modeling #####

m_norm <- lm(DI_scaled ~ Group, data = TS)

assumptions(m_norm)

# Adjust beta distribution

m_beta <- glmmTMB(DI_scaled ~ Group, data = TS,
                family = beta_family(link = "logit"))

dharma <- simulateResiduals(fittedModel = m_beta)
plot(dharma)


# Model heterocedasticity

m_beta.2 <- glmmTMB(DI_scaled ~ Group, data = TS,
                dispformula = ~ Group,
                family = beta_family(link = "logit"))

dharma.2 <- simulateResiduals(fittedModel = m_beta.2)
plot(dharma.2)


#### AIC ####

AIC(m_norm, m_beta, m_beta.2)


#### Results ####

model <- m_beta

summary(model)
Anova(model)


# Interaction comparisons

comp <- emmeans(model, pairwise ~ Group)


# Final Plot

plot(comp$emmeans, comparisons = TRUE)


# Extract summary measures

model_sum <- as.data.frame(comp$emmeans)
model_sum


# confidence bands

model_plot <- ggpredict(model, terms = c("Group"), interval = "confidence")   
model_plot

#

plot(model_plot, show_data = T, jitter = 0.1) + 
  ggtitle("Predicted values") + 
  labs(y = "DI", colour = "Group")

