import mesa
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Choice
import json

from agents.bacolod_model import BacolodModel
from agents.household_agent import HouseholdAgent
from agents.enforcement_agent import EnforcementAgent
from agents.barangay_agent import BarangayAgent

# =========================================================================
# THE MONKEY PATCH: Intercepting Custom WebSocket Commands
# =========================================================================
from mesa.visualization.ModularVisualization import SocketHandler

original_on_message = SocketHandler.on_message

def custom_on_message(self, message):
    try:
        msg = json.loads(message)
        if msg.get("type") == "switch_view":
            print(f"\n[SUCCESS] Web UI requested map switch to: {msg['value']}")
            if hasattr(self.application, 'model'):
                self.application.model.current_view = msg["value"]
            return  
    except Exception as e:
        print(f"WebSocket Error: {e}")
        
    original_on_message(self, message)

SocketHandler.on_message = custom_on_message
# =========================================================================


# --- 1. The Phantom Portrayal ---
def dynamic_portrayal(agent):
    if agent is None: return None
    
    target_bgy = getattr(agent.model, 'current_view', "BGY_0")
    b_id = getattr(agent, 'barangay_id', None)
    if b_id is None:
        b_id = getattr(agent, 'unique_id', None) 
        
    if b_id != target_bgy: 
        return None

    portrayal = {"Filled": "true"}
    agent_class = agent.__class__.__name__ 
    
    if agent_class == "HouseholdAgent":
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.5   
        portrayal["Layer"] = 0
        portrayal["Color"] = "green" if getattr(agent, "is_compliant", False) else "red"
        
    elif agent_class == "EnforcementAgent":
        is_mun = getattr(agent, "is_municipal", False)
        portrayal["Shape"] = "rect"
        portrayal["Layer"] = 1
        portrayal["w"] = 0.9 if is_mun else 0.7
        portrayal["h"] = 0.9 if is_mun else 0.7
        portrayal["Color"] = "purple" if is_mun else "blue"
        portrayal["text"] = "M" if is_mun else "T"
        portrayal["text_color"] = "white"
        
    elif agent_class == "BarangayAgent":
        portrayal["Shape"] = "circle"
        portrayal["r"] = 1.0  
        portrayal["Layer"] = 2
        portrayal["Color"] = "black"
        
    return portrayal

# --- 2. Live Map Switcher & Chart Labeler ---
class ViewSwitcher(mesa.visualization.TextElement):
    def __init__(self):
        self.first_render = True

    def render(self, model):
        if not self.first_render:
            return ""
        
        self.first_render = False
        
        return """
        <div id="stable-ui-container" style="width: 100%; text-align: center; margin-bottom: 20px;">
            <div style="padding: 15px; background: #ffffff; border-radius: 8px; border: 2px solid #28a745; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #333;">Live Barangay View Selector</h4>
                <select id="live-bgy-select" style="padding: 8px 24px; font-size: 16px; font-weight: bold; border-radius: 4px; border: 1px solid #ccc; cursor: pointer; background: #f8f9fa;" onchange="updatePythonView()">
                    <option value="BGY_0">Poblacion</option>
                    <option value="BGY_1">Liangan East</option>
                    <option value="BGY_2">Ezperanza</option>
                    <option value="BGY_3">Binuni</option>
                    <option value="BGY_4">Demologan</option>
                    <option value="BGY_5">Mati</option>
                    <option value="BGY_6">Babalaya</option>
                </select>
            </div>
        </div>

        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" style="display:none;" onload="
            if (!window.uiDetached) {
                window.uiDetached = true;
                
                // 1. Move Dropdown
                setTimeout(() => {
                    let ui = document.getElementById('stable-ui-container');
                    let elementsDiv = document.getElementById('elements'); 
                    if (ui && elementsDiv) {
                        elementsDiv.insertBefore(ui, elementsDiv.firstChild);
                    }
                }, 100);

                // 2. WebSocket Sender
                window.updatePythonView = function() {
                    let selectEl = document.getElementById('live-bgy-select');
                    let selected = selectEl.value;
                    
                    selectEl.style.borderColor = '#28a745';
                    setTimeout(() => { selectEl.style.borderColor = '#ccc'; }, 200);

                    let msgObj = {'type': 'switch_view', 'value': selected};
                    
                    try {
                        if (typeof send === 'function') {
                            send(msgObj);
                        } else if (window.ws && window.ws.readyState === 1) {
                            window.ws.send(JSON.stringify(msgObj));
                        }
                    } catch (e) {
                        console.error('WebSocket communication failed:', e);
                    }
                };

                // 3. Foolproof Chart Labeler
                setInterval(() => {
                    let canvases = document.getElementsByTagName('canvas');
                    
                    // Only run if we have both the map canvas and the chart canvas
                    if (canvases.length > 1) {
                        // The chart is always the LAST canvas rendered by Mesa
                        let chartCanvas = canvases[canvases.length - 1];
                        let chartContainer = chartCanvas.parentElement;
                        
                        if(!document.getElementById('thesis-y-axis')) {
                            chartContainer.style.position = 'relative';
                            chartContainer.style.padding = '20px 20px 60px 80px'; 
                            chartContainer.style.marginTop = '30px';
                            
                            let yLabel = document.createElement('div');
                            yLabel.id = 'thesis-y-axis';
                            yLabel.innerHTML = 'Compliance Rate (%)';
                            yLabel.style.position = 'absolute';
                            yLabel.style.left = '-15px'; // Pushed left slightly so it doesn't overlap numbers
                            yLabel.style.top = '84%';
                            yLabel.style.transform = 'translateY(-50%) rotate(-90deg)';
                            yLabel.style.fontWeight = 'bold';
                            yLabel.style.fontSize = '16px';
                            yLabel.style.color = '#333';
                            chartContainer.appendChild(yLabel);
                            
                            let xLabel = document.createElement('div');
                            xLabel.id = 'thesis-x-axis';
                            xLabel.innerHTML = 'Ticks (Days)';
                            xLabel.style.position = 'absolute';
                            xLabel.style.bottom = '10px';
                            xLabel.style.left = '50%';
                            xLabel.style.transform = 'translateX(-50%)';
                            xLabel.style.fontWeight = 'bold';
                            xLabel.style.fontSize = '16px';
                            xLabel.style.color = '#333';
                            chartContainer.appendChild(xLabel);
                        }
                    }
                }, 1000);
            }
        ">
        """

# --- 3. Setup Elements ---
visual_elements = [ViewSwitcher()]

grid = CanvasGrid(dynamic_portrayal, 50, 50, 600, 600)
visual_elements.append(grid)

barangay_chart_data = [
    {"Label": "Brgy Poblacion",    "Color": "red"},     
    {"Label": "Brgy Liangan East", "Color": "orange"},  
    {"Label": "Brgy Ezperanza",    "Color": "gold"},    
    {"Label": "Brgy Binuni",       "Color": "green"},   
    {"Label": "Brgy Babalaya",     "Color": "cyan"},    
    {"Label": "Brgy Mati",         "Color": "blue"},    
    {"Label": "Brgy Demologan",    "Color": "purple"}   
]

chart_compliance = ChartModule(
    [{"Label": "Global Compliance", "Color": "Black"}] + barangay_chart_data,
    data_collector_name='datacollector',
    canvas_width=1000, 
    canvas_height=400 
)
visual_elements.append(chart_compliance)

# --- 4. Launch ---
model_params = {
    "seed": 42,
    "train_mode": False,
    "policy_mode": Choice(
        name="LGU Policy Strategy",  
        value="status_quo",      
        choices=["NO_LGU", "status_quo", "pure_incentives", "pure_enforcement", "HuDRL"]
    ),
    "current_view": "BGY_0" 
}

server = ModularServer(
    BacolodModel,
    visual_elements,
    "Bacolod Multi-Layered Governance Simulation",
    model_params
)

server.port = 8522 
server.launch()