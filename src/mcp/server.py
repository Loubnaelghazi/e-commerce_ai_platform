import asyncio
import json
import uuid
import websockets
from typing import Dict, Callable, Any
import time

from src.mcp.protocol import MCPMessage, MessageType



  
class MCPServer:

    def __init__(self, agent_name: str, port: int = 8000):
        self.agent_name = agent_name
        self.port = port
        self.tools: Dict[str, Callable] = {}
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        
    #####################    
    def add_tool(self, name: str, func: Callable):

        self.tools[name] = func
        print(f"[{self.agent_name}] Tool registered: {name}")



    ############
    async def handle_client(self, websocket, path):
        """Gère les connexions client"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        print(f"[{self.agent_name}] Client connected: {client_id}")
        
        try:
            async for message_str in websocket:
                await self.process_message(message_str, client_id)
        except websockets.exceptions.ConnectionClosed:
            print(f"[{self.agent_name}] Client disconnected: {client_id}")

        finally:
            del self.clients[client_id]


    ##################
    async def process_message(self, message_str: str, client_id: str):
        """Traite les messages reçus"""
        try:
            message_data = json.loads(message_str)
            message = MCPMessage(
                id=message_data['id'],
                type=MessageType(message_data['type']),
                payload=message_data['payload'],
                sender=message_data['sender'],
                receiver=message_data['receiver'],
                timestamp=message_data['timestamp'])
            
            if message.type == MessageType.TOOL_CALL:
                await self.handle_tool_call(message, client_id)
                
        except Exception as e:
            await self.send_error(client_id, str(e))
    
    ########################
    async def handle_tool_call(self, message: MCPMessage, client_id: str):
        """Exécute un appel d'outil"""
        tool_name = message.payload.get('tool_name')
        tool_args = message.payload.get('args', {})
        
        if tool_name not in self.tools:
            await self.send_error(client_id, f"Tool not found: {tool_name}")
            return
        
        try:
            result = await self.tools[tool_name](tool_args)
            
            response = MCPMessage(
                id=str(uuid.uuid4()),
                type=MessageType.TOOL_RESPONSE,
                payload={'result': result, 'original_id': message.id},
                sender=self.agent_name,
                receiver=message.sender,
                timestamp=time.time() )
            
            await self.clients[client_id].send(response.to_json())
            
        except Exception as e:
            await self.send_error(client_id, f"Tool execution error: {str(e)}")


    ##########################
    async def send_error(self, client_id: str, error_msg: str):
        """Envoie un message d'erreur"""
        error = MCPMessage(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR,
            payload={'error': error_msg},
            sender=self.agent_name,
            receiver="unknown",
            timestamp=time.time()
        )
        await self.clients[client_id].send(error.to_json())
    




    ###########################
    async def start(self):
        """Demarre le serveur MCP"""
        
        print(f"[{self.agent_name}] Starting MCP server on port {self.port}")
        return await websockets.serve(self.handle_client, "localhost", self.port)
