# Graphic and statistic analysis of Object Recognition Tasks in mice.

# Introduction -----------------------------------------------------------------

#### Setup ####

ls()
rm(list=ls())
ls()

# libraries

Dir <- "C:/Users/dhers/OneDrive - UBA/Seguimiento"
setwd(Dir)

list.of.packages <- c("ggplot2", "reshape2", "corrplot", "nlme", "emmeans", "multcomp", "car", "dplyr", "doBy", "Rmisc", "ggeffects", "sjPlot", "lme4", "glmmTMB", "DHARMa", "pastecs", "psych", "effects", "margins", "GGally", "gridExtra", "MuMIn", "performance",  "lmerTest", "Hmisc", "readxl", "ggsignif")

# install.packages("easypackages")
easypackages::packages(list.of.packages)


#### Load the data ####

TORM_2m_3h <- read_excel("Resultados para Seguimiento.xlsx", sheet = "TORM 2m 3h")
TORM_3m_3h <- read_excel("Resultados para Seguimiento.xlsx", sheet = "TORM 3m 3h")
TORM_3m_24h <- read_excel("Resultados para Seguimiento.xlsx", sheet = "TORM 3m 24h")
TORM_2m_24h <- read_excel("Resultados para Seguimiento.xlsx", sheet = "TORM 2m 24h")

TORM <- rbind(TORM_2m_3h, TORM_3m_3h, TORM_3m_24h, TORM_2m_24h)

class(TORM)     # tipo de Object
str(TORM)       # estructura

TORM$Mouse <- factor(TORM$Mouse)
TORM$Box <- factor(TORM$Box)
TORM$Home <- factor(TORM$Home)
TORM$Litter <- factor(TORM$Litter)
TORM$Mark <- factor(TORM$Mark)
TORM$Sex <- factor(TORM$Sex, levels = c("Female", "Male"))
TORM$Age <- factor(TORM$Age, levels = c("2 mth", "3 mth"))
TORM$Use <- factor(TORM$Use)
TORM$Group <- factor(TORM$Group)
TORM$Wait <- factor(TORM$Wait, levels = c("3 hs", "24 hs"))
TORM$Trial <- factor(TORM$Trial, levels = c("TR1", "TR2", "TS"))
TORM$Side <- factor(TORM$Side)
TORM$Object <- factor(TORM$Object)
TORM$Novelty <- factor(TORM$Novelty, levels = c("Recent", "Old"))

str(TORM)
dim(TORM)       # dimensiones
head(TORM)      # primeras 6 filas (se pueden pedir mas filas si lo desean)
summary(TORM)   # resuemn de las variables

TORM

# Subsetting (me quedo solo con TS)

TS <- subset(TORM, Trial == "TS")
summary(TS)

outliers <- c(1.06, 3.18)  # Add all the values you want to remove
TS <- TS[!TS$Mouse %in% outliers, ]

# Grafico

ggplot(TS, aes(x = Novelty, y = Time, fill = Novelty)) + 
  geom_boxplot(outlier.size = 2, outlier.shape = 4, width = 0.5) + # Outliers
  scale_fill_manual(values = c("gray", "goldenrod1")) +
  stat_summary(fun = "mean", geom = "point", shape = 16, size = 2) +
  geom_jitter(data = TS, width = 0.2, size = 1.5, aes(color = Sex, shape = Sex)) +
  geom_line(aes(group = Mouse, color = Sex), linewidth = 1, alpha = 0.25) +
  # geom_hline(aes(yintercept=0), colour="grey", size=1.5, alpha = 0.8) + # Línea en 0
  ggtitle("Exploration of old and recent familiar objects") + # Título
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("Time (s)") + theme(axis.title.y = element_text(angle = 90)) +
  theme(axis.title.x = element_blank()) + # Saco título de los ejes
  stat_summary(fun = "mean", geom = "point", shape = 16, size = 2) + 
  facet_grid(Sex ~ Age*Wait)
  # scale_y_continuous(limits = c(0, 45))

# Calculo el ID

for_ID <- dcast(TS, Mouse + Home + Litter + Mark + Sex + Age + Use + Group + Wait + Trial ~ Novelty, value.var = "Time")

for_ID$ID <- ((for_ID$Old - for_ID$Recent) / (for_ID$Old + for_ID$Recent)) * 100

for_ID

# Grafico

ggplot(for_ID, aes(x = interaction(Sex), y = ID)) + 
  geom_boxplot(outlier.size = 2, outlier.shape = 4, width = 0.5) + # Outliers
  # scale_fill_manual(values = c("gray", "goldenrod1")) +
  stat_summary(fun = "mean", geom = "point", shape = 16, size = 2) +
  geom_jitter(data = for_ID, width = 0.2, size = 1.5, aes(color = Sex, shape = Sex)) +
  geom_line(aes(group = Mouse, color = Sex), linewidth = 1, alpha = 0.25) +
  geom_hline(aes(yintercept=0), colour="grey", size=1.5, alpha = 0.8) + # Línea en 0
  ggtitle("Object recognition during TS") + # Título
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("ID (%)") + 
  theme(axis.title.x = element_blank()) + # Saco título de los ejes
  # scale_y_continuous(limits = c(-30, 80)) + 
  facet_grid( ~ Age*Wait)


ggplot(for_ID, aes(x = interaction(Rol, Trial), y = ID, fill = Trial)) + geom_boxplot(outlier.colour = NA) + # Qué graficar
  geom_jitter(width = 0.1, size = 1.5,aes(color = Sex, shape = Sex)) + # Puntos de colores
  ggtitle("Exploración de cada objeto en entrenamientos y testeo") + # Título
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("Time de exploración (s)") + theme(axis.title.y = element_text(angle = 90)) +
  theme(axis.title.x = element_blank()) + # Saco título de los ejes
  facet_grid( ~ Age*Time)


# Grafico separando por Home (veo la variación entre Mousees y membranas)

TS2 <- subset(Comp2, Trial == "TS")

summary(TS2)

ggplot(TS2, aes(x = interaction(Rol, Trial), y = Exploracion, fill = Rol)) + geom_boxplot(outlier.colour = NA) + # Qué graficar
  scale_fill_manual(values = c("dimgray", "darkgoldenrod1")) +
  geom_jitter(width = 0.1, size = 1.5,aes(color = Sex, shape = Sex)) + # Puntos de colores
  ggtitle("Exploración de cada Object en el testeo") + # Título
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("Time de exploración (s)") + theme(axis.title.y = element_text(angle = 90)) +
  theme(axis.title.x = element_blank()) + # Saco título de los ejes
  facet_grid(Sex ~ Age*Time)


#### Funcion supuestos ####

supuestos <- function(modelo) {
  
  residuos <- resid(modelo, type = "pearson")
  predichos <- predict(modelo)
  
  par(mfrow = c(1, 2))
  
  plot(x = predichos,
       y = residuos,
       # ylim = c(-4, 4),
       xlab = "Predichos",
       ylab = "Residuos de Pearson",
       main = "Grafico de dispersion de residuos v. predichos", 
       cex.main = 0.8 )
  
  abline(h = c(-2, 2, 0),
         col = c("red", "red", "black"),
         lty = c(2, 2, 1))
  
  #  qqnorm(residuos, cex.main = 0.8)
  #  qqline(residuos)
  qqPlot(residuos)
  
  return(list(
    shapiro.test(residuos),
    leveneTest(residuos, predichos)))
  
}

# MuMIn

dredge(lm(ID ~ Trial*Age*Time*Sex, na.action = "na.fail", data = Comp))

# El ajuste es óptimo cuando considero la interacción entre el grupo y la estructura cerebral.


##### Estadística #####

# Aditivo

m0 <- lm(ID ~ Trial + Age + Time + Sex, 
         data = Comp)

supuestos(m0)


# Con interacciones

m1 <- lm(ID ~ Trial*Age*Time*Sex, 
         data = Comp)

supuestos(m1)


#### Declaro la falta de independencia ####


# con lme4

m2 <- lmer(ID ~ Trial*Age*Time*Sex +
             (1|Home/Mouse) + (1|Litter),
             data = Comp)

# Estoy tomando al ratón anidado en su Home de proveniencia como variables aleatorias, así como la membrana de incubación.

supuestos(m2)


# modelando la varianza ####

# relación media-varianza

media <- matrix(tapply(Comp$ID, Comp$Home, mean))
varianza <- matrix(tapply(Comp$ID, Comp$Home, var))
sd  <- matrix(tapply(Comp$ID, Comp$Home, sd))

plot(x = media,
     y = varianza,
     main = "var vs mean - Home",
     ylab = "varianza",
     col = "red")

media2 <- matrix(tapply(Comp$ID, Comp$Object, mean))
varianza2 <- matrix(tapply(Comp$ID, Comp$Object, var))
sd2  <- matrix(tapply(Comp$ID, Comp$Object, sd))

plot(x = media2,
     y = varianza2,
     main = "var vs mean - Object",
     ylab = "varianza",
     col = "blue")

# Veo una relación lineal entre la varianza y la media... 

# lme no me deja poner dos variables aleatorias, qué hago con la membrana?


# varIdent

m3.0 <- lme(ID ~ Trial*Age*Time*Sex,
            random = ~ 1|Home/Mouse,
            weights = varIdent(form = ~ 1|Home),
            data = Comp)

supuestos(m3.0)


# varPower

m3.1 <- lme(ID ~ Trial*Age*Time*Sex,
            random = ~ 1|Home/Mouse,
            weights = varPower(),
            data = Comp)

supuestos(m3.1) 


# varPower

m3.2 <- lme(ID ~ Trial*Age*Time*Sex,
            random = ~ 1|Home/Mouse,
            weights = varExp(),
            data = Comp)

supuestos(m3.2) 


# Lo mejor es estimar una varianza para cada grupo experimental?


#### Con glmmTMB ####

m4.0 <- glmmTMB(ID ~ Trial*Age*Time*Sex,
                data = Comp)

dharma.0 <- simulateResiduals(fittedModel = m4.0)
plot(dharma.0)


# declaro falta de independencia

m4.1 <- glmmTMB(ID ~ Trial*Age*Time*Sex + 
                (1|Home/Mouse),
                data = Comp)

dharma.1 <- simulateResiduals(fittedModel = m4.1)
plot(dharma.1)


# modelo la varianza

m4.2 <- glmmTMB(ID ~ Trial*Age*Time*Sex + 
                  (1|Home/Mouse),
                dispformula = ~ Home, 
                data = Comp)

dharma.2 <- simulateResiduals(fittedModel = m4.2)
plot(dharma.2)


# ajusto a distribución beta

m4.3 <- glmmTMB(ID ~ Trial*Age*Time*Sex + 
                (1|Home/Mouse),
                data = Comp,
                family = beta_family(link = "logit"))

dharma.3 <- simulateResiduals(fittedModel = m4.3)
plot(dharma.3)


# modelo la varianza

m4.4 <- glmmTMB(ID ~ Trial*Age*Time*Sex + 
                  (1|Home/Mouse),
                dispformula = ~ Home, 
                data = Comp,
                family = beta_family(link = "logit"))

dharma.4 <- simulateResiduals(fittedModel = m4.4)
plot(dharma.4)


#### AIC ####

AIC(m0, m1, m2, m4.0, m4.1, m4.2, m4.3, m4.4)


#### Resultados ####

modelo <- m2

summary(modelo)
Anova(modelo)

drop1(modelo, test = "Chisq") # El modelo mejora si no evalúo la interacción con el Sex, pero no es significativo


# Resultado: Interaccion significativa entre Grupo y Parte

# Evalúo igual la interacción con la fracción?


# seteo el emmeans

emm_options(emmeans = list(infer = c(TRUE, FALSE)),
            contrast = list(infer = c(TRUE, TRUE)))


# Comparaciones de interacción

comp <- emmeans(modelo, pairwise ~ Trial|Time*Age*Sex) #Tukey por default
comp


# Grafico final

plot(comp$emmeans, comparisons = TRUE)


# Extraemos las medidas resumen

resumen_modelo <- as.data.frame(comp$emmeans)
resumen_modelo


# Bandas de confianza ####

model_plot <- ggpredict(modelo, 
                        terms = c("Trial", "Time", "Sex"),
                        interval = "confidence")   
model_plot

plot(model_plot, add.data = T) + 
  ggtitle("valores predichos") + 
  labs(y = "ID", colour = "Time")

