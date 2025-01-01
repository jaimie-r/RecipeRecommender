from sklearn.model_selection import train_test_split
import numpy as np
import requests
import pandas as pd
import re
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Your Spoonacular API key
API_KEY = '2672152fccaa4240b436feb8ecf71bc2'

# Function to get detailed recipe information
def get_recipes_from_api(query, recipes, num_results=10):
    url = f'https://api.spoonacular.com/recipes/complexSearch?query={query}&number={num_results}&apiKey={API_KEY}'
    response = requests.get(url)

    if response.status_code == 402:
        print("API limit reached.")
        return False
              
    data = response.json()

    # Check for valid results
    if 'results' not in data or not data['results']:
        print("No results found for query:", query)
        return True  # Continue fetching
    
    # Extracting the relevant details from the response
    for recipe in data['results']:
        recipe_id = recipe['id']  # Get recipe ID for detailed information
        
        # Fetching detailed recipe information
        detailed_url = f'https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={API_KEY}'
        detailed_response = requests.get(detailed_url)
        detailed_data = detailed_response.json()

        if not detailed_data:
            print(f"Error: Empty JSON response for recipe_id {recipe_id}")
            continue

        if response.status_code == 402:
            print("API limit reached.")
            return False

        if 'title' not in detailed_data:
            print(f"Missing 'title' in the response for recipe_id {recipe_id}")
            print("Full response:", detailed_data)  # Print the full response for debugging
            return False  # Skip this recipe

        recipe_name = detailed_data['title']

        # Check if the recipe name is already in the list
        if any(recipe[0] == recipe_name for recipe in recipes):
            continue  # Skip duplicates

        ingredients = detailed_data['extendedIngredients']
        # Combine ingredient names into a single text string
        ingredient_text = " ".join([ingredient['name'] for ingredient in ingredients])

        number_of_servings = detailed_data['servings']

        # Extracting nutritional information
        summary = detailed_data['summary']  # Extracting the summary field for nutrition

        # Use regex to extract protein from the summary
        protein = None
        calories = None

        # Regex to match protein from the summary text
        protein_match = re.search(r'(\d+)\s*g of protein', summary)
        calories_match = re.search(r'(\d+)\s* calories', summary)

        if protein_match:
            protein = int(protein_match.group(1))
        if calories_match:
            calories = int(calories_match.group(1))

        protein_density = 0
        if calories > 0:
            protein_density = protein / calories

        # Estimate the total number of vegetables in the recipe
        vegetable_ingredients = [ing for ing in ingredients if ing['aisle'] == 'Produce']
        num_vegetable_ingredients = len(vegetable_ingredients)

        recipes.append([
            recipe_name,
            ingredient_text,
            number_of_servings,
            calories,
            protein,
            protein_density,
            num_vegetable_ingredients
        ])

    return True

# Function to load the recipe data from CSV or API
def load_recipes_data():
    if os.path.exists('data/recipes_with_embeddings.csv'):
        # Load the data from CSV if it exists
        df = pd.read_csv('data/recipes_with_embeddings.csv')
        print("Data loaded from CSV.")
    else:
        # Fetch the recipes using the API and save them to CSV
        queries = ["fries", "chicken", "steak", "sausage", "bacon", "burger", "cheese", "potato", 
            "pepperoni", "ramen", "tortilla", "bowl", "cauliflower", "kale", "salmon", "tuna", 
            "quinoa", "lentil"]
        recipes = []
        for query in queries:
            if not get_recipes_from_api(query, recipes):
                break

        # Create a DataFrame
        df = pd.DataFrame(recipes, columns=[
            'recipe_name', 'ingredient_text', 'number_of_servings',
            'calories', 'protein', 'protein_density', 'num_vegetable_servings'
        ])

        # Save to CSV
        df.to_csv('data/recipes_with_embeddings.csv', index=False)
        print("Data fetched and saved to CSV.")
    
    return df

# Load the recipes data
df = load_recipes_data()

# Load pretrained GloVe embeddings (50-dimensional vectors)
def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_model[word] = vector
    return glove_model

# Load the GloVe model (ensure the path to the file is correct)
glove_model = load_glove_model('/Users/jaimieren/Downloads/glove.6B/glove.6B.50d.txt') 

# Function to get the average embedding for a list of ingredients
def get_average_embedding(ingredients, glove_model):
    embeddings = []
    for ingredient in ingredients.split():
        if ingredient in glove_model:
            embeddings.append(glove_model[ingredient])
        else:
            # If the ingredient isn't in the vocabulary, use a zero vector
            embeddings.append(np.zeros(50))  # Assuming 50-dimensional vectors
    return np.mean(embeddings, axis=0)

# Create embeddings for each recipe
df['ingredient_embedding'] = df['ingredient_text'].apply(lambda x: get_average_embedding(x, glove_model))

# weighted sum of protein density and vegetable servings for target variable
protein_density_weight = 2
vegetable_servings_weight = 1

df['target'] = (protein_density_weight * df['protein_density']) + (vegetable_servings_weight * df['num_vegetable_servings'])

# Save to CSV
df.to_csv('data/recipes_with_embeddings.csv', index=False)

# Prepare the feature matrix (X) and target variable (y)

# Nutritional features (e.g., calories, protein, etc.)
nutritional_features = df[['protein_density', 'num_vegetable_servings']].values

# Ingredient embeddings (assumes you have them already)
ingredient_embeddings = np.array(df['ingredient_embedding'].tolist())

# Concatenate the nutritional features and ingredient embeddings
X = np.concatenate([nutritional_features, ingredient_embeddings], axis=1)

y = df['target'] 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsRegressor(n_neighbors=5)  # Adjust 'n_neighbors' as needed
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

def get_user_input():
    """
    Prompts the user for the ingredients, protein density, and number of vegetable servings.
    """
    # Get ingredients as a list
    ingredients_input = input("Enter ingredients (separate by commas): ")
    ingredients = [ingredient.strip() for ingredient in ingredients_input.split(",")]

    # Get protein density
    protein_density_input = input("Enter protein density: ")
    protein_density = float(protein_density_input)

    # Get number of vegetable servings
    num_vegetable_servings_input = input("Enter number of vegetable servings: ")
    num_vegetable_servings = int(num_vegetable_servings_input)

    return ingredients, protein_density, num_vegetable_servings

def recommend_recipe(ingredients, protein_density, num_vegetable_servings, glove_model, df, knn):
    """
    Recommends a recipe based on input ingredients and nutritional values.

    Parameters:
        ingredients (list of str): List of ingredient names.
        protein_density (float): Protein density value.
        num_vegetable_servings (int): Number of vegetable servings.
        glove_model (dict): Pretrained GloVe embeddings.
        df (DataFrame): Original dataset with recipe details.
        knn (KNeighborsRegressor): Trained KNN model.

    Returns:
        dict: Recommended recipe details with protein density and vegetable servings.
    """
    # Compute the average embedding for the input ingredients
    embeddings = []
    for ingredient in ingredients:
        if ingredient in glove_model:
            embeddings.append(glove_model[ingredient])
        else:
            # Use a zero vector for unknown words
            embeddings.append(np.zeros(50))  # Assuming 50-dimensional embeddings
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = np.zeros(50)  # Default embedding for empty input

    # Combine nutritional features and ingredient embedding
    nutritional_features = [protein_density, num_vegetable_servings]
    input_features = np.concatenate([nutritional_features, avg_embedding]).reshape(1, -1)

    # Predict the nearest neighbor
    distances, indices = knn.kneighbors(input_features, n_neighbors=1)
    nearest_index = indices[0][0]

    # Fetch the corresponding recipe details
    recommended_recipe = df.iloc[nearest_index]
    return {
        "recipe_name": recommended_recipe["recipe_name"],
        "protein_density": recommended_recipe["protein_density"],
        "num_vegetable_servings": recommended_recipe["num_vegetable_servings"]
    }

def recommend_recipe_from_input(glove_model, df, knn):
    """
    Runs the recommendation system after receiving user input.
    """
    # Get input from the user
    ingredients, protein_density, num_vegetable_servings = get_user_input()

    # Recommend the recipe based on the input
    recommendation = recommend_recipe(ingredients, protein_density, num_vegetable_servings, glove_model, df, knn)

    # Display the recommendation
    print("\nRecommended Recipe:")
    print(f"Name: {recommendation['recipe_name']}")
    print(f"Protein Density: {recommendation['protein_density']}")
    print(f"Number of Vegetable Servings: {recommendation['num_vegetable_servings']}")

# Run the recommendation system
recommend_recipe_from_input(glove_model, df, knn)
