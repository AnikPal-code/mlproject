def initiate_model_trainer(self, train_array, test_array):
    try:
        logging.info("Split training and test input data")
        print("Splitting data...")  # Add this print statement

        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )
        print("Data split complete.")  # Add this print statement

        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }
        params = {
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Linear Regression": {},
            "XGBRegressor": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "AdaBoost Regressor": {
                'learning_rate': [.1, .01, 0.5, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        print("Starting model evaluation...")  # Add this print statement

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, param=params)
        
        print("Model evaluation complete.")  # Add this print statement
        
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No best model found")

        logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")
        print(f"Best found model: {best_model_name} with score: {best_model_score}")  # Add this print statement

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )

        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)

        print(f"R2 Score: {r2_square}")  # Add this print statement
        return r2_square

    except Exception as e:
        print(f"An error occurred: {e}")  # Add this print statement
        raise CustomException(e, sys)
