from IPython.display import Image, display
from IPython.display import clear_output, display
import requests
from io import BytesIO
from PIL import Image as PILImage

import logging
import requests
from typing import Optional
from agentscope.service.service_response import ServiceResponse, ServiceExecStatus
from agentscope.service.service_toolkit import ServiceToolkit
import agentscope
from agentscope.message import Msg
from agentscope.agents.dialog_agent import DialogAgent as BaseDialogAgent
from agentscope.agents import DialogAgent
import json
import ast

from typing import Dict, Any, List
from agentscope.parsers import ParserBase
from agentscope.models import ModelResponse
import re
import os

class LocationMatcherParser(ParserBase):
    """A parser to match locations in natural language text against a predefined list of locations."""

    def __init__(self, locations):
        """
        Initialize the parser with a list of location dictionaries.
        
        Args:
            locations (list): A list of dictionaries, each containing location information.
        """
        self.locations = locations

    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Parse the response text to find matches with the predefined locations.
        
        Args:
            response (ModelResponse): The response object containing the text to parse.
        
        Returns:
            ModelResponse: The response object with the parsed result added.
        """
        text = response.text
        matches = []

        for location in self.locations:
            if re.search(r'\b' + re.escape(location['name']) + r'\b', text, re.IGNORECASE):
                matches.append(location)

        response.parsed = matches
        return response
    
import agentscope
agentscope.init(
    # ...
    project="xxx",
    name="xxx",
    studio_url="http://127.0.0.1:5000"          # The URL of AgentScope Studio
)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agentscope.service import (
    get_tripadvisor_location_photos,
    search_tripadvisor,
    get_tripadvisor_location_details
)

# Initialize the ServiceToolkit and register the TripAdvisor API functions
service_toolkit = ServiceToolkit()
service_toolkit.add(search_tripadvisor, api_key="")  # Replace with your actual TripAdvisor API key
service_toolkit.add(get_tripadvisor_location_details, api_key="")  # Replace with your actual TripAdvisor API key
service_toolkit.add(get_tripadvisor_location_photos, api_key="")  # Replace with your actual TripAdvisor API key


class ExtendedDialogAgent(BaseDialogAgent):
    def __init__(self, name: str, sys_prompt: str, model_config_name: str, use_memory: bool = True, memory_config: Optional[dict] = None):
        super().__init__(name, sys_prompt, model_config_name, use_memory, memory_config)
        self.service_toolkit = service_toolkit
        self.current_location = None
        self.current_details = None
        self.game_state = "start"
    
    def parse_service_response(self, response: str) -> dict:
        """
        Parse the service response string and extract the JSON content.
        Args:
            response (str): The response string from the service call.
        Returns:
            dict: The parsed JSON content.
        """
        result = {}
        lines = response.split('\n')
        for line in lines:
            if '[STATUS]:' in line:
                status = line.split('[STATUS]:')[-1].strip()
                result['status'] = ServiceExecStatus.SUCCESS if status == 'SUCCESS' else ServiceExecStatus.ERROR
            if '[RESULT]:' in line:
                json_str = line.split('[RESULT]:')[-1].strip()
                result['content'] = ast.literal_eval(json_str)
        return result

    def propose_location(self):
        messages = [
            {"role": "system", "content": "You are a game master for a geography guessing game. Propose an interesting location as for the player to guess. Diversify your proposal and give less known ones."},
            {"role": "user", "content": "Propose an interesting location for the player to guess. Only output your final proposed location."}
        ]
        response = self.model(messages).text.strip()
        return response

    def search_and_select_location(self, proposed_location):
        response_str = self.service_toolkit.parse_and_call_func(
            [{"name": "search_tripadvisor", "arguments": {"query": proposed_location}}]
        )
        result = self.parse_service_response(response_str)
        if result['status'] == ServiceExecStatus.SUCCESS and result['content']['data']:
            locations = result['content']['data']
            if len(locations) > 1:
                messages = [
                    {"role": "system", "content": "You are selecting the most appropriate location from a list of search results."},
                    {"role": "user", "content": f"Select the location that best matches '{proposed_location}' from this list:\n" + 
                    "\n".join([f"{i+1}. {loc['name']}, {loc.get('address_obj', {}).get('city', 'Unknown City')}, {loc.get('address_obj', {}).get('country', 'Unknown Country')}" for i, loc in enumerate(locations)]) +
                    "\nRespond with only the number of your selection."}
                ]
                selection_response = self.model(messages).text.strip()
                try:
                    # Try to extract the number from the response
                    selection = int(''.join(filter(str.isdigit, selection_response))) - 1
                    if 0 <= selection < len(locations):
                        return locations[selection]
                    else:
                        # If the extracted number is out of range, return the first location
                        return locations[0]
                except ValueError:
                    # If no number can be extracted, return the first location
                    return locations[0]
            else:
                return locations[0]
        return None


    def get_location_details(self, location_id):
        response_str = self.service_toolkit.parse_and_call_func(
            [{"name": "get_tripadvisor_location_details", "arguments": {"location_id": location_id}}]
        )
        return self.parse_service_response(response_str)
    
    def get_location_photos(self, location_id: str) -> dict:
        """
        Get the largest photo for a specific location.
        
        Args:
            location_id (str): The location ID to get photos for.
        
        Returns:
            dict: The largest photo information including the URL.
        """
        logger.info(f"Calling TripAdvisor API for location photos: {location_id}")
        response_str = self.service_toolkit.parse_and_call_func(
            [{"name": "get_tripadvisor_location_photos", "arguments": {"location_id": location_id}}]
        )
        logger.info(f"TripAdvisor location photos result: {response_str}")
        result = self.parse_service_response(response_str)
        return result['content']['largest_photo'] if result['status'] == ServiceExecStatus.SUCCESS else None
    

    def display_photo(self, location_id):
        photos = self.get_location_photos(location_id)
        if photos:
            try:
                image_url = photos['url']
                response = requests.get(image_url)
                img = PILImage.open(BytesIO(response.content))
                display(img)
                return True
            except Exception as e:
                logger.error(f"Error displaying image: {str(e)}")
        return False

    # def check_guess(self, user_guess, ancestors):
    #     for ancestor in ancestors:
    #         if user_guess.lower() in ancestor['name'].lower():
    #             return ancestor['level'], ancestor['name']
    #     return None, None

    def check_guess_location(self, user_guess, location):
 
        messages = [
        {"role": "system", "content": '''You are the gamemaster of a geoguessr game. Check if the player's guess: '''+ str(user_guess)+''' means the same place/location as '''+ str(location)+'''. If yes, return 'True'. Else, return 'False'. Only output one of the two options given and nothing else.'''},
        {"role": "user", "content": user_guess}
        ]

        response = self.model(messages).text.strip()
        logger.info(f"check_guess: {response}")
        return response

            
    
    def check_guess_ancestors(self, user_guess, ancestors):
        # parser = LocationMatcherParser(ancestors)
        # response = ModelResponse(text="This location is an indigenous village nestled within a picturesque canyon in the southwestern part of the United States. It's accessible via a trail that winds through the rugged landscape of Arizona.")

        # parsed_response = parser.parse(response)

        matches = []

        for location in ancestors:
            if re.search(r'\b' + re.escape(location['name']) + r'\b', user_guess, re.IGNORECASE):
                matches.append(location)
        if not matches:
            return 'None','None','False'
        else:
            messages = [
            {"role": "system", "content": '''Check if '''+ str(matches)+''' include the smallest level in '''+ str(ancestors)+''' based on their respective levels. If yes, return the smallest matched name and the corresponding level as 'level,name,True'. Else, return 'False'. Note that the placeholders 'level', 'name' in the output are to be replaced by the actual values. Only output one of the two options given and nothing else.'''},
            {"role": "user", "content": user_guess}
            ]
        # messages = [
        #     {"role": "system", "content": '''You're a game master for a geography guessing game. Check user's guess if any part of the guess (including Country) matches any name at any level in ancestors: '''+ str(ancestors)+'''. If doesn't match at all levels, return 'None,None,False'. 
        #      If it matches but the smallest matched name's corresponding level is not the smallest level available in ancestors, return the smallest **matched** name and the corresponding level as 'level,name,False'.
        #      If it matches and the smallest matched name's corresponding level is the smallest level available in ancestors, return the smallest matched name and the corresponding level as 'level,name,True'.
        #      Only output one of the three options given and nothing else.'''},
        #     {"role": "user", "content": user_guess}
        # ]
            response = self.model(messages).text.strip()
            if 'True' in response:
                logger.info(f"check_guess: {response}")
                response = response.split(',')
                return response[0],response[1],response[2]
            else:
                messages = [
                {"role": "system", "content": '''Return the smallest name based on their respective 'level' values and the corresponding 'level' values in '''+ str(matches)+''' as 'level,name,False'. Note that the placeholders 'level', 'name' in the output are to be replaced by the actual values. Only output in the given format and nothing else.'''},
                {"role": "user", "content": user_guess}
                ]
                response = self.model(messages).text.strip()
                logger.info(f"check_guess: {response}")
                response = response.split(',')
                return response[0],response[1],response[2]

    def save_image_from_url(self, url, save_path):
        try:
            os.makedirs(save_path, exist_ok=True)
            # Send a HTTP request to the URL
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check if the request was successful

            # Get the file name from the URL
            file_name = os.path.basename(url)
            file_name = 'image'+os.path.splitext(file_name)[1]
            # Full path to save the image
            full_path = os.path.join(save_path, file_name)

            # Open a local file with write-binary mode
            with open(full_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Image saved successfully at {full_path}")
            return full_path
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_hint(self, details):
        messages = [
            {"role": "system", "content": "You're a game master for a geography guessing game."},
            {"role": "user", "content": f"give the user some hint about the location based on the info from {details}. Don't make it too obvious."}
        ]
        response = self.model(messages).text.strip()
        logger.info(f"Hint: {response}")
        return response


    def handle_input(self, user_input: dict) -> dict:
        query = user_input['text']

        if self.game_state == "start":
            photo_displayed = False
            while not photo_displayed:
                proposed_location = self.propose_location()
                self.current_location = self.search_and_select_location(proposed_location)
                print('self.current_location: ', self.current_location)
                if not self.current_location:
                    return {"text": "I'm sorry, I couldn't find a suitable location. Let's try again."}
                
                self.current_details = self.get_location_details(self.current_location['location_id'])
                if self.current_details['status'] != ServiceExecStatus.SUCCESS:
                    return {"text": "I'm having trouble getting details for this location. Let's try again."}
                ancestors = self.current_details['content'].get('ancestors', [])
                print('ancestors: ', ancestors)

                # clear_output(wait=True)
                display(None)
                photo_displayed = self.display_photo(self.current_location['location_id'])
                photos = self.get_location_photos(self.current_location['location_id'])
                
            response = "Let's play a geography guessing game! I've chosen a location and displayed an image of it. "
            # response += f"![image]({photos['url']})"
            response += " Can you guess which country, state, region, city, or municipality this location is in?"
            self.game_state = "guessing"
            image_path = self.save_image_from_url(photos['url'], save_path = "./images")

            return [{"text": response}, {"image": image_path}]

        elif self.game_state == "guessing":
            if self.check_guess_location(query.lower(), self.current_location['name'].lower()) == 'True':
                self.game_state = "end"
                return {"text": f"Congratulations! You've guessed correctly. The location is indeed in {self.current_location['name']}."}
            ancestors = self.current_details['content'].get('ancestors', [])
            level, correct_name, is_smallest = self.check_guess_ancestors(query, ancestors)

            if level != 'None':
                if is_smallest == 'True':
                    self.game_state = "end"
                    return {"text": f"Congratulations! You've guessed correctly. The location is indeed in {level}: {correct_name}."}
                else:
                    return {"text": f"Good guess! {level}: {correct_name} is correct, but can you be more specific? Try guessing a smaller region or city."}
            else:
                hint = self.get_hint(self.current_details['content'])
                return {"text": f"I'm sorry, that's not correct. Here's a hint: {hint} Try guessing again!"}

        else:
            return {"text": "I'm sorry, I don't understand. Let's start a new game!"}

# Initialize AgentScope and run
def main() -> None:
    """A GeoGuessr-like game demo"""

    agentscope.init(
        model_configs=[
            {
                "model_type": "openai_chat",
                "config_name": "gpt-3.5-turbo",
                "model_name": "gpt-3.5-turbo",
                "api_key": "",  # Load from env if not provided
                "generate_args": {
                    "temperature": 0.5,
                },
            },
            {
                "config_name": "dashscope_chat-qwen-max",
                "model_type": "dashscope_chat",
                "model_name": "qwen-max-1201",
                "api_key": "",
                "generate_args": {
                    "temperature": 0.1
                }
            },
            {
                "config_name": "dashscope_multimodal-qwen-vl-max",
                "model_type": "dashscope_multimodal",
                "model_name": "qwen-vl-max",
                "api_key": "",
                "generate_args": {
                    "temperature": 0.01
                }
            },
            {
                "config_name": "gpt-4o_config",
                "model_type": "openai_chat",
                "model_name": "gpt-4o",
                "api_key": "",
                "generate_args": {
                    "temperature": 0.8,
            },
        }
        ],
    )

    # Init the dialog agent
    gamemaster_agent = ExtendedDialogAgent(
        name="GeoGuessr Gamemaster",
        sys_prompt="You're a game master for a geography guessing game.",
        model_config_name="gpt-4o_config",
        use_memory=True,
    )

    player_agent = DialogAgent(
        name="player",
        sys_prompt='''You're a player in a geoguessr-like turn-based game. Upon getting an image from the gamemaster, you are supposed to guess where is the place shown in the image. Your guess can be a country,
        a state, a region, a city, etc., but try to be as precise as possoble. If your answer is not correct, try again based on the hint given by the gamemaster.''',
        model_config_name="dashscope_multimodal-qwen-vl-max",  # replace by your model config name
    )

    # Start the game
    x = None
    while gamemaster_agent.game_state != 'guessing':    
        
        response = gamemaster_agent.handle_input({"text": "Let's start the game"})
        x = Msg("GeoGuessr Gamemaster", response[0]['text'], url=response[1]['image'])
        
        # x = Msg("GeoGuessr Gamemaster", response)
        gamemaster_agent.speak(x)
        print("Game Master:", response[0]['text'])

    # Main game loop
    
    while gamemaster_agent.game_state != 'end':
        # Clear previous output
        # clear_output(wait=True)
        
        # Display the image and game master's text
        # if gamemaster_agent.game_state == "guessing":
        #     gamemaster_agent.display_photo(gamemaster_agent.current_location['location_id'])
        # print("Game Master:", response['text'])
        
        # Force display update
        display(None)
        
        # user_input = input("Your guess: ")
        x = player_agent(x)
        if 'quit' in x['content'].lower():
            print("Thanks for playing!")
            break
        response = gamemaster_agent.handle_input({"text": x['content']})
        x = Msg("GeoGuessr Gamemaster", response['text'])
        gamemaster_agent.speak(x)
        print("Game Master:", response['text'])


    #pve
    
    # # Start the game
    # x = None
    # while gamemaster_agent.game_state != 'guessing':    
        
    #     response = gamemaster_agent.handle_input({"text": "Let's start the game"})
    #     x = Msg("GeoGuessr Gamemaster", response['text'])

    #     print("Game Master:", response['text'])

    # # Main game loop
    
    # while gamemaster_agent.game_state != 'end':
    #     # Clear previous output
    #     # clear_output(wait=True)
        
    #     # Display the image and game master's text
    #     # if gamemaster_agent.game_state == "guessing":
    #     #     gamemaster_agent.display_photo(gamemaster_agent.current_location['location_id'])
    #     print("Game Master:", response['text'])
        
    #     # Force display update
    #     display(None)
        
    #     user_input = input("Your guess: ")
    #     if user_input.lower() == 'quit':
    #         print("Thanks for playing!")
    #         break
    #     response = gamemaster_agent.handle_input({"text": user_input})
    #     print("Game Master:", response['text'])

if __name__ == "__main__":
    main()