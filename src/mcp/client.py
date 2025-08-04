from typing import Any, Dict
import uuid
import websockets
import json
import asyncio
import time

from mcp.protocol import MCPMessage, MessageType


'''Permet à un agent de se connecter à un autre agent via WebSocket et d'appeler des "outils" (fonctions) à distance sur cet agent.'''
''' notes :
Si tu n’es pas connecté à Agent_n, tu ne peux pas appeler l’outil.

Si Agent_n ne repond pas dans les 30 secondes, tu as une erreur timeout.

Si une erreur arrive, tu peux l’attraper et agir.


'''

class MCPClient:
    def __init__(self, agent_name: str):

        self.agent_name = agent_name
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.pending_calls: Dict[str, asyncio.Future] = {}
    




    async def connect(self, target_agent: str, uri: str):
        """Se connecte à un autre agent"""
        try:
            websocket = await websockets.connect(uri)
            self.connections[target_agent] = websocket
            
            print(f"[{self.agent_name}] Connected to {target_agent}")
            
            asyncio.create_task(self.listen_responses(target_agent))
            
        except Exception as e:
            print(f"[{self.agent_name}] Failed to connect to {target_agent}: {e}")


    
    async def call_tool(self, target_agent: str, tool_name: str, args: Dict[str, Any]) -> Any:

        """ Appelle un outil sur un autre agent """

        if target_agent not in self.connections:
            raise Exception(f"Not connected to {target_agent}")
        
        call_id = str(uuid.uuid4())
        message = MCPMessage(
            id=call_id,
            type=MessageType.TOOL_CALL,
            payload={'tool_name': tool_name, 'args': args},
            sender=self.agent_name,
            receiver=target_agent,
            timestamp=time.time() )
        
        future = asyncio.Future()
        self.pending_calls[call_id] = future
        
        await self.connections[target_agent].send(message.to_json())
        
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            del self.pending_calls[call_id]
            raise Exception(f"Timeout calling {tool_name} on {target_agent}")
    




    async def listen_responses(self, target_agent: str):
        websocket = self.connections[target_agent]
        try:
            async for message_str in websocket:
                message_data = json.loads(message_str)
                
                if message_data['type'] == MessageType.TOOL_RESPONSE.value:
                    original_id = message_data['payload']['original_id']
                    if original_id in self.pending_calls:
                        future = self.pending_calls[original_id]
                        future.set_result(message_data['payload']['result'])
                        del self.pending_calls[original_id]
                        
        except websockets.exceptions.ConnectionClosed:
            print(f"[{self.agent_name}] Connection to {target_agent} closed")
        except Exception as e:
            print(f"[{self.agent_name}] Error listening to {target_agent}: {e}")