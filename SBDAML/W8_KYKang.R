# 패키지 설치 및 로드
#install.packages(c("sf", "spData", "spdep", "caret", "blockCV", "randomForest", "dplyr"))
library(sf)
library(spData)
library(spdep)
library(caret)
library(blockCV)
library(randomForest)
library(dplyr)

# 1. 공간 데이터 로드, S2 비활성화, 유효성 보정, 평면 좌표계 변환
sf::sf_use_s2(FALSE)
data(world, package = "spData")
world_sf <- st_as_sf(world)
world_sf <- st_make_valid(world_sf)
world_sf <- st_transform(world_sf, 3857)

# 2. 공간 가중치 행렬 생성 (4-최근접 이웃)
coords <- st_coordinates(st_centroid(world_sf))
nb <- knn2nb(knearneigh(coords, k = 4))
listw <- nb2listw(nb, style = "W")

# 3. listw 객체를 행렬로 변환하고 중심화 행렬 생성
W <- listw2mat(listw)
n <- nrow(W)
I <- diag(n)
one <- matrix(1, n, n)
M <- I - (1/n) * one

# 4. 중심화된 공간 가중치 행렬 MWM 계산 & 대칭 고윳값 분해
MWM <- M %*% W %*% M
eig <- eigen(MWM, symmetric = TRUE)

# 5. 양의 고윳값에 해당하는 고유벡터 (공간 필터) 추출
pos_idx <- which(eig$values > 0)
pos_ev <- eig$vectors[, pos_idx]

# 6. ESF 변수 10개 추출
esf_vars <- as.data.frame(pos_ev[, 1:10])
colnames(esf_vars) <- paste0("ESF_", 1:10)

# 7. 원본 데이터에 ESF 변수 결합
world_esf <- cbind(world_sf, esf_vars)

# 8. 결측치 및 pop <= 0 제거
world_esf <- world_esf %>% filter(!is.na(pop) & pop > 0)

# 9. SKCV용 공간 블록 생성 (원본 변수용)
set.seed(123)
sb <- spatialBlock(speciesData = world_esf,
                   theRange = 500000,
                   k = 5,
                   selection = "random",
                   showBlocks = FALSE)
world_esf$fold <- sb$foldID

# 10. ESF 변수용 일반 K-Fold 교차검증용 trainControl 설정
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)

# 11. ESF 변수 이용 랜덤포레스트: caret 일반 K-Fold CV
df_esf <- st_drop_geometry(world_esf)[, c(paste0("ESF_", 1:10), "pop")]
rf_esf_caret <- train(log(pop) ~ ., data = df_esf,
                      method = "rf",
                      trControl = train_control)
cat("ESF 변수 일반 K-Fold CV 결과:\n")
print(rf_esf_caret)

# 12. SKCV (공간 블록 K-Fold)로 원본 변수(gdpPercap) 랜덤포레스트 평가
rmse_list_skcv <- c()
for (i in 1:5) {
  train_data <- world_esf[world_esf$fold != i, ]
  test_data <- world_esf[world_esf$fold == i, ]
  
  train_data <- train_data %>% filter(!is.na(pop), pop > 0, !is.na(gdpPercap))
  test_data <- test_data %>% filter(!is.na(pop), pop > 0, !is.na(gdpPercap))
  
  train_df <- st_drop_geometry(train_data)
  test_df <- st_drop_geometry(test_data)
  
  rf_model <- randomForest(log(pop) ~ gdpPercap, data = train_df)
  
  pred <- predict(rf_model, newdata = test_df)
  rmse <- sqrt(mean((pred - log(test_df$pop))^2))
  rmse_list_skcv <- c(rmse_list_skcv, rmse)
  
  cat(paste0("SKCV Fold ", i, " RMSE (gdpPercap): ", round(rmse, 4), "\n"))
}

cat("SKCV 평균 RMSE (원본 변수):", round(mean(rmse_list_skcv), 4), "\n")

# 13. ⭐ 원본 변수(gdpPercap) 이용한 일반 K-Fold 교차검증 (caret)
df_basic <- st_drop_geometry(world_esf) %>% 
  filter(!is.na(pop), pop > 0, !is.na(gdpPercap)) %>%
  select(pop, gdpPercap)

rf_basic_caret <- train(log(pop) ~ gdpPercap,
                        data = df_basic,
                        method = "rf",
                        trControl = train_control)
cat("\n원본 변수 일반 K-Fold CV 결과 (gdpPercap):\n")
print(rf_basic_caret)


# =========================================
# (이전 코드 동일)
# =========================================
# 1~13까지 동일하게 실행한 후 아래 코드 추가
# =========================================

# 14. SKCV (공간 블록 K-Fold)로 ESF 변수(Random Forest) 평가
rmse_list_esf_skcv <- c()

for (i in 1:5) {
  train_data <- world_esf[world_esf$fold != i, ]
  test_data  <- world_esf[world_esf$fold == i, ]
  
  # 결측치 처리
  train_data <- train_data %>% filter(!is.na(pop), pop > 0)
  test_data  <- test_data %>% filter(!is.na(pop), pop > 0)
  
  # ESF 변수만 추출
  train_df <- st_drop_geometry(train_data)[, c(paste0("ESF_", 1:10), "pop")]
  test_df  <- st_drop_geometry(test_data)[, c(paste0("ESF_", 1:10), "pop")]
  
  # 랜덤포레스트 모델 학습
  rf_model_esf <- randomForest(log(pop) ~ ., data = train_df)
  
  # 예측 및 RMSE 계산
  pred <- predict(rf_model_esf, newdata = test_df)
  rmse <- sqrt(mean((pred - log(test_df$pop))^2))
  rmse_list_esf_skcv <- c(rmse_list_esf_skcv, rmse)
  
  cat(paste0("SKCV Fold ", i, " RMSE (ESF 변수): ", round(rmse, 4), "\n"))
}

cat("SKCV 평균 RMSE (ESF 변수):", round(mean(rmse_list_esf_skcv), 4), "\n")

# =========================================
# 결과 요약 출력
# =========================================
cat("\n\n==== 모델별 교차검증 결과 요약 ====\n")
cat("1 ESF 변수 (일반 K-Fold / caret): RMSE =", 
    round(rf_esf_caret$results$RMSE[which.min(rf_esf_caret$results$RMSE)], 4), "\n")
cat("2 원본 변수 (일반 K-Fold / caret): RMSE =", 
    round(rf_basic_caret$results$RMSE[which.min(rf_basic_caret$results$RMSE)], 4), "\n")
cat("3 원본 변수 (SKCV): 평균 RMSE =", 
    round(mean(rmse_list_skcv), 4), "\n")
cat("4 ESF 변수 (SKCV): 평균 RMSE =", 
    round(mean(rmse_list_esf_skcv), 4), "\n")