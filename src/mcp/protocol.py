from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import uuid


'''

 MCP n’est qu’un protocole de message JSON simple, avec :

    tool_call pour demander.

    tool_response pour repondre.

'''



'''
{
  "type": "tool_call",
  "id": "123....",  
  "tool_name": "name_exp",
  "parameters": { "x": 5, "y": 2 ..... }
}


'''


'''
msgs response exp :

{
  "type": "tool_response",
  "id": "123",
  "output": 7
}


'''

class MessageType(Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class MCPMessage:
    id: str
    type: MessageType
    payload: Dict[str, Any]
    sender: str
    receiver: str
    timestamp: float

    def to_json(self) -> str:
        return json.dumps({
            'id': self.id,
            'type': self.type.value,
            'payload': self.payload,
            'sender': self.sender,
            'receiver': self.receiver,
            'timestamp': self.timestamp
        })
