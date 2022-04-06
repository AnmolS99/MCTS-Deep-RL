# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient
from hex_actor import HexActor

actor = HexActor(k=7, model_path="models/model_k7_300_of_400")


class MyClient(ActorClient):

    def handle_get_action(self, state):
        row, col = actor.get_action(state)  # Your logic
        return row, col


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth="aa5c0269e3254d9a95675779c36852ef", qualify=False)
    client.run()