#Step 3: Visualize data
plt.scatter(df["hours_studied"], df["exam_marks"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Mark")
plt.title("Study Hours vs Exam Marks")
plt.show()

#step 4 : Define features and targety
X = df[["hours_studied"]]
y = df["exam_marks"]

#step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=42
)
print("Training data:")
print(X_train)
print("Test data")
print(X_test)

# Step 6: Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Prediction on test data
y_pred = model.predict(X_test)
print("Actual marks:", list(y_test))
print("Predicted marks:", list(y_pred))

# Step 8: Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 9: Predict for new value
new_hours = [[9]]
predicted_mark = model.predict(new_hours)
print(f"If a student studies 9 hours, predicted marks: {predicted_mark[0]:.2f}")

# Step 10: Plot regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Marks")
plt.title("Linear Regression")
plt.legend()
plt.show()
