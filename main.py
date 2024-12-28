from sklearn.model_selection import train_test_split
import numpy as np
import requests
import pandas as pd
import re
import os
import json

# Your Spoonacular API key
API_KEY = '2672152fccaa4240b436feb8ecf71bc2'

# Function to get detailed recipe information
def get_recipes_from_api(query, recipes, num_results=5):
    url = f'https://api.spoonacular.com/recipes/complexSearch?query={query}&number={num_results}&apiKey={API_KEY}'
    response = requests.get(url)

    if response.status_code == 402:
        print("API limit reached.")
        return

    # Check for the presence of the 'results' key
    if 'results' not in response or not response['results']:
        print("Error: No results found in the API response.")
        return
              
    data = response.json()
    
    # Extracting the relevant details from the response
    for recipe in data['results']:
        recipe_id = recipe['id']  # Get recipe ID for detailed information
        
        # Fetching detailed recipe information
        detailed_url = f'https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={API_KEY}'
        detailed_response = requests.get(detailed_url)
        detailed_data = detailed_response.json()

        if not detailed_data:
            print(f"Error: Empty JSON response for recipe_id {recipe_id}")
            return
        
        # print(json.dumps(detailed_data, indent=2))
        # print("**************")

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

# Function to load the recipe data from CSV or API
def load_recipes_data():
    if os.path.exists('data/recipes_with_embeddings.csv'):
        # Load the data from CSV if it exists
        df = pd.read_csv('data/recipes_with_embeddings.csv')
        print("Data loaded from CSV.")
    else:
        # Fetch the recipes using the API and save them to CSV
        query = "burger"
        recipes = []
        for i in range(20):
            get_recipes_from_api(query, recipes)
            query = recipes[i][0].split(" ")[0]  # Update query for next API call

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
nutritional_features = df[['calories', 'protein', 'protein_density', 'num_vegetable_servings']].values

# Ingredient embeddings (assumes you have them already)
ingredient_embeddings = np.array(df['ingredient_embedding'].tolist())

# Concatenate the nutritional features and ingredient embeddings
X = np.concatenate([nutritional_features, ingredient_embeddings], axis=1)

y = df['target'] 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your model (for example, a regression model) could go here
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)