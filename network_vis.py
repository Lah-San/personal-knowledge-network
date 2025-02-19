import os
import numpy as np
import networkx as nx
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import json
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import numpy as np
from tqdm import tqdm
import sys

@dataclass
class Node3D:
    id: str
    position: np.ndarray
    weight: float
    color: Tuple[float, float, float]
    related_nodes: Set[str]

class TagNetworkProcessor:
    def __init__(self, base_tag_dir: str):
        self.base_tag_dir = base_tag_dir
        self.embeddings = OllamaEmbeddings(
            model="olmo2",
            base_url="http://localhost:11434"
        )
        
    def get_tag_directories(self) -> List[str]:
        """Get all tag directories"""
        return [d for d in os.listdir(self.base_tag_dir) 
                if os.path.isdir(os.path.join(self.base_tag_dir, d))]
    
    def load_tag_store(self, tag: str) -> Chroma:
        """Load the vector store for a specific tag"""
        tag_dir = os.path.join(self.base_tag_dir, tag)
        return Chroma(
            persist_directory=tag_dir,
            embedding_function=self.embeddings
        )
    
    def calculate_tag_weight(self, tag: str) -> float:
        """Calculate tag weight based on number of documents"""
        try:
            store = self.load_tag_store(tag)
            return len(store.get()['ids'])
        except Exception as e:
            print(f"Error calculating weight for tag {tag}: {str(e)}")
            return 1.0
    
    def calculate_tag_similarity(self, tag1: str, tag2: str) -> float:
        """Calculate similarity between two tags"""
        try:
            store1 = self.load_tag_store(tag1)
            store2 = self.load_tag_store(tag2)
            
            # Get embeddings
            embeds1 = store1.get()['embeddings']
            embeds2 = store2.get()['embeddings']
            
            if not embeds1 or not embeds2:
                return 0.0
            
            # Calculate average similarity
            similarities = []
            for embed1 in embeds1[:5]:  # Limit to first 5 documents for performance
                for embed2 in embeds2[:5]:
                    similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
                    similarities.append(similarity)
            
            return float(np.mean(similarities))
        except Exception as e:
            print(f"Error calculating similarity between {tag1} and {tag2}: {str(e)}")
            return 0.0
    
    def generate_network_data(self) -> Dict:
        """Generate network data from tag directories"""
        print("Processing tag directories...")
        tags = self.get_tag_directories()
        
        # Calculate tag weights
        print("Calculating tag weights...")
        nodes = []
        for tag in tqdm(tags):
            weight = self.calculate_tag_weight(tag)
            nodes.append({"id": tag, "weight": weight})
        
        # Calculate tag relationships
        print("Calculating tag relationships...")
        edges = []
        for i, tag1 in enumerate(tqdm(tags)):
            for tag2 in tags[i+1:]:
                similarity = self.calculate_tag_similarity(tag1, tag2)
                if similarity > 0.3:  # Only keep significant relationships
                    edges.append({
                        "source": tag1,
                        "target": tag2,
                        "weight": similarity
                    })
        
        return {"nodes": nodes, "edges": edges}

class TagNetwork3DVisualizer:
    def __init__(self, network_data: Dict):
        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(1024, 768)
        glutCreateWindow(b"Tag Network 3D Visualizer")

        # Enable OpenGL features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)

        # Process network data
        self.process_network_data(network_data)
        
        # View parameters
        self.camera_distance = 20.0
        self.camera_rotation = [30.0, 0.0]
        self.last_mouse = None
        self.selected_node = None
        self.zoom_speed = 1.0
        
        # Color mapping for relationships
        self.relationship_colors = {
            'strong': (0.0, 0.8, 0.0),    # Green
            'medium': (0.8, 0.8, 0.0),    # Yellow
            'weak': (0.8, 0.0, 0.0)       # Red
        }
        
        # Register callbacks
        glutDisplayFunc(self.display)
        glutMouseFunc(self.mouse_click)
        glutMotionFunc(self.mouse_motion)
        glutMouseWheelFunc(self.mouse_wheel)
        glutKeyboardFunc(self.keyboard)
        glutReshapeFunc(self.reshape)
        
        # Initialize node positions
        self.initialize_node_positions()

    def process_network_data(self, data: Dict):
        """Process network data into internal format"""
        self.nodes: Dict[str, Node3D] = {}
        self.edges: List[Tuple[str, str, float]] = []
        
        # Normalize weights
        weights = [node['weight'] for node in data['nodes']]
        min_weight = min(weights)
        max_weight = max(weights)
        weight_range = max_weight - min_weight
        
        # Process nodes
        for node in data['nodes']:
            norm_weight = 0.5 + 1.5 * (node['weight'] - min_weight) / weight_range if weight_range > 0 else 1.0
            
            related_nodes = {edge['target'] for edge in data['edges'] 
                           if edge['source'] == node['id']}
            related_nodes.update({edge['source'] for edge in data['edges'] 
                                if edge['target'] == node['id']})
            
            self.nodes[node['id']] = Node3D(
                id=node['id'],
                position=np.zeros(3),
                weight=norm_weight,
                color=(0.2, 0.5, 1.0),
                related_nodes=related_nodes
            )
        
        # Process edges
        self.edges = [(edge['source'], edge['target'], edge['weight']) 
                     for edge in data['edges']]

    def initialize_node_positions(self):
        """Initialize node positions using force-directed layout"""
        G = nx.Graph()
        
        for node_id in self.nodes:
            G.add_node(node_id)
        for source, target, weight in self.edges:
            G.add_edge(source, target, weight=weight)
        
        pos = nx.spring_layout(G, dim=3, k=2.0)
        
        for node_id, position in pos.items():
            self.nodes[node_id].position = position * 10.0

    def reshape(self, width: int, height: int):
        """Handle window reshape"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width/height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def draw_sphere(self, node: Node3D, radius: float, color: Tuple[float, float, float]):
        """Draw a sphere for a node with its label"""
        position = node.position
        
        glPushMatrix()
        glTranslatef(*position)
        
        # Draw sphere
        glColor4f(*color, 1.0)
        quad = gluNewQuadric()
        gluSphere(quad, radius, 32, 32)
        
        # Always draw label
        if self.selected_node and (node.id == self.selected_node.id or 
                                 node.id in self.selected_node.related_nodes):
            glColor4f(1.0, 1.0, 1.0, 1.0)  # White for highlighted
            scale = 0.015
        else:
            glColor4f(0.7, 0.7, 0.7, 0.7)  # Gray for normal
            scale = 0.01
        
        # Draw label
        glPushMatrix()
        glTranslatef(0, radius + 0.2, 0)
        glRotatef(-self.camera_rotation[1], 0, 1, 0)
        glRotatef(-self.camera_rotation[0], 1, 0, 0)
        glScalef(scale, scale, scale)
        for c in node.id:
            glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, ord(c))
        glPopMatrix()
        
        glPopMatrix()

    def get_relationship_color(self, weight: float) -> Tuple[float, float, float]:
        """Get color based on relationship weight"""
        if weight >= 0.7:
            return self.relationship_colors['strong']
        elif weight >= 0.4:
            return self.relationship_colors['medium']
        else:
            return self.relationship_colors['weak']

    def draw_edge(self, start: np.ndarray, end: np.ndarray, 
                  color: Tuple[float, float, float], alpha: float, weight: float):
        """Draw an edge between two points"""
        relationship_color = self.get_relationship_color(weight)
        
        if self.selected_node:
            glColor4f(*color, alpha)
        else:
            glColor4f(*relationship_color, 0.4)
            
        glLineWidth(max(1, weight * 5))
        
        glBegin(GL_LINES)
        glVertex3f(*start)
        glVertex3f(*end)
        glEnd()
        
        glLineWidth(1)

    def draw_legend(self):
        """Draw relationship strength legend"""
        glPushMatrix()
        
        # Reset transformations for 2D overlay
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
        # Draw legend background
        glColor4f(0.2, 0.2, 0.2, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(200, 10)
        glVertex2f(200, 100)
        glVertex2f(10, 100)
        glEnd()
        
        # Draw legend items
        y = 30
        for label, color in [
            ("Strong Relationship", self.relationship_colors['strong']),
            ("Medium Relationship", self.relationship_colors['medium']),
            ("Weak Relationship", self.relationship_colors['weak'])
        ]:
            glColor4f(*color, 1.0)
            glLineWidth(3)
            glBegin(GL_LINES)
            glVertex2f(20, y)
            glVertex2f(50, y)
            glEnd()
            
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glRasterPos2f(60, y + 5)
            for c in label:
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))
            
            y += 20
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def display(self):
        """Main display function"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set up camera
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_rotation[0], 1, 0, 0)
        glRotatef(self.camera_rotation[1], 0, 1, 0)
        
        # Draw edges
        for source, target, weight in self.edges:
            source_node = self.nodes[source]
            target_node = self.nodes[target]
            
            if self.selected_node and (source == self.selected_node.id or 
                                     target == self.selected_node.id):
                color = (1.0, 0.5, 0.0)
                alpha = 1.0
            else:
                color = self.get_relationship_color(weight)
                alpha = 0.4
            
            self.draw_edge(source_node.position, target_node.position, 
                          color, alpha, weight)
        
        # Draw nodes
        for node in self.nodes.values():
            if self.selected_node:
                if node.id == self.selected_node.id:
                    color = (1.0, 0.0, 0.0)
                elif node.id in self.selected_node.related_nodes:
                    color = (1.0, 0.5, 0.0)
                else:
                    color = (0.5, 0.5, 0.5)
            else:
                color = node.color
            
            self.draw_sphere(node, 0.3 * node.weight, color)
        
        # Draw legend when no node is selected
        if not self.selected_node:
            self.draw_legend()
        
        glutSwapBuffers()

    def get_node_at_cursor(self, cursor_x: int, cursor_y: int) -> Node3D:
        """Return the node at the cursor position"""
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        cursor_y = viewport[3] - cursor_y
        z = glReadPixels(cursor_x, cursor_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        
        cursor_pos = gluUnProject(cursor_x, cursor_y, z[0][0], 
                                modelview, projection, viewport)
        
        min_dist = float('inf')
        closest_node = None
        
        for node in self.nodes.values():
            dist = np.linalg.norm(node.position - np.array(cursor_pos))
            if dist < min_dist and dist < 1.0:
                min_dist = dist
                closest_node = node
        
        return closest_node

    def mouse_click(self, button: int, state: int, x: int, y: int):
        """Handle mouse clicks"""
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.selected_node = self.get_node_at_cursor(x, y)
                self.last_mouse = (x, y)
            else:
                self.last_mouse = None
        glutPostRedisplay()

    def mouse_motion(self, x: int, y: int):
        """Handle mouse motion"""
        if self.last_mouse:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            
            self.camera_rotation[1] += dx * 0.5
            self.camera_rotation[0] += dy * 0.5
            
            # Clamp vertical rotation
            self.camera_rotation[0] = max(-90, min(90, self.camera_rotation[0]))
            
            self.last_mouse = (x, y)
            glutPostRedisplay()

    def mouse_wheel(self, wheel: int, direction: int, x: int, y: int):
        """Handle mouse wheel for zooming"""
        self.camera_distance += -direction * self.zoom_speed
        self.camera_distance = max(5.0, min(50.0, self.camera_distance))
        glutPostRedisplay()

    def keyboard(self, key: bytes, x: int, y: int):
        """Handle keyboard input"""
        if key == b'\x1b':  # ESC key
            sys.exit(0)
        elif key == b'r':  # Reset view
            self.camera_distance = 20.0
            self.camera_rotation = [30.0, 0.0]
            self.selected_node = None
        glutPostRedisplay()

    def run(self):
        """Start the visualization"""
        glutMainLoop()

def main():
    """Main function to run the visualization"""
    # Default directory if none provided
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database/tags")
    
    # Get base directory from command line or use default
    if len(sys.argv) > 1:
        base_tag_dir = sys.argv[1]
    else:
        base_tag_dir = default_dir
        if not os.path.exists(base_tag_dir):
            os.makedirs(base_tag_dir)
            print(f"Created default tag directory at: {base_tag_dir}")
    
    # Verify directory exists
    if not os.path.exists(base_tag_dir):
        print(f"Error: Directory not found: {base_tag_dir}")
        print(f"Please create the directory or specify a valid path.")
        sys.exit(1)
    
    # Process tag network
    try:
        processor = TagNetworkProcessor(base_tag_dir)
        print("Generating network data...")
        network_data = processor.generate_network_data()
        
        # Save network data
        with open('network_data.json', 'w') as f:
            json.dump(network_data, f)
        print("Network data saved to network_data.json")
        
        # Create and run visualizer
        visualizer = TagNetwork3DVisualizer(network_data)
        print("Starting visualization...")
        visualizer.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()