from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain




# Recipe recommendation template.
template = """
You are a helpful assistant that recommends Nigerian recipes. Use the following context to provide a recipe recommendation.
If you can't find a recipe that matches the ingredients in the context, just say you don't have enough information.
Do not make up any recipes.

Context: {context}

Based on the following ingredients: {ingredients}.
The recipe should be {cuisine} and have {flavor_profile} flavors.

Please help me by:

* **Suggesting a unique dish name:** {dish_name}
* **Providing a list of ingredients:** {ingredients}
* **Creating detailed instructions:** {instructions}
"""


# The 'input_variables' tell the template what values it needs to fill in.
prompt = PromptTemplate(
    input_variables=["ingredients", "cuisine", "flavor_profile"],
    template=template,
)

# Initialize the VertexAI model.
llm = VertexAI(model_name="gemini-1.5-pro-preview-0514")

chain = LLMChain(llm=llm, prompt=prompt)

def get_recipe(ingredients: str, cuisine: str, flavor_profile: str) -> str:
    """
    Generates a recipe based on a list of ingredients, a cuisine, and a flavor profile.

    Args:
        ingredients (str): A comma-separated list of ingredients.
        cuisine (str): The desired cuisine (e.g., "Nigerian").
        flavor_profile (str): The desired flavor profile (e.g., "spicy and savory").

    Returns:
        str: A string containing the formatted recipe.
    """
    try:
        # Run the chain to get the recipe.
        # The 'generate' method will fill in the template variables and
        # get the model's response.
        response = chain.invoke(
            {
                "ingredients": ingredients,
                "cuisine": cuisine,
                "flavor_profile": flavor_profile,
            }
        )
        return response["text"]
    except Exception as e:
        return f"An error occurred: {e}"
