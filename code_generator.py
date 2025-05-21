import math

class ArduinoCodeGenerator:
    def __init__(self):
        """Initialize the Arduino code generator"""
        self.turn_speed = 90  # degrees per command
        self.forward_speed = 100  # mm per command
        self.delay_duration = 500  # ms
    
    def generate_movement_commands(self, path, grid_path):
        """
        Generate a list of movement commands from the path
        
        Args:
            path: List of (x, y) coordinates representing the path
            grid_path: List of grid cells in the path
            
        Returns:
            commands: List of dictionaries with movement commands
        """
        if len(path) < 2:
            return []
        
        commands = []
        current_direction = None
        
        # Calculate initial direction
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        
        if abs(dx) > abs(dy):
            current_direction = 'E' if dx > 0 else 'W'
        else:
            current_direction = 'S' if dy > 0 else 'N'
        
        # Process each segment in the path
        for i in range(1, len(path)):
            prev_point = path[i-1]
            curr_point = path[i]
            
            # Calculate segment direction
            dx = curr_point[0] - prev_point[0]
            dy = curr_point[1] - prev_point[1]
            
            # Determine new direction
            new_direction = None
            if abs(dx) > abs(dy):
                new_direction = 'E' if dx > 0 else 'W'
            else:
                new_direction = 'S' if dy > 0 else 'N'
            
            # If direction changed, add a turn command
            if new_direction != current_direction:
                turn_angle = self._calculate_turn_angle(current_direction, new_direction)
                if turn_angle != 0:
                    turn_type = 'R' if turn_angle > 0 else 'L'
                    commands.append({
                        'type': turn_type,
                        'angle': abs(turn_angle),
                        'description': f"Turn {'right' if turn_type == 'R' else 'left'} {abs(turn_angle)} degrees"
                    })
                    
                    # Add delay after turn
                    commands.append({
                        'type': 'D',
                        'duration': self.delay_duration,
                        'description': f"Delay {self.delay_duration}ms"
                    })
                
                current_direction = new_direction
            
            # Calculate distance to move
            distance = math.sqrt(dx**2 + dy**2) * self.forward_speed / 10  # Scale based on maze size
            
            # Add forward command
            commands.append({
                'type': 'F',
                'duration': int(distance),
                'description': f"Move forward {int(distance)}mm"
            })
            
            # Add delay after movement
            commands.append({
                'type': 'D',
                'duration': int(self.delay_duration / 5),  # Shorter delay after forward movement
                'description': f"Delay {int(self.delay_duration / 5)}ms"
            })
        
        return commands
    
    def _calculate_turn_angle(self, from_dir, to_dir):
        """
        Calculate the turn angle between two directions
        
        Args:
            from_dir: Current direction (N, E, S, W)
            to_dir: New direction (N, E, S, W)
            
        Returns:
            angle: Turn angle in degrees (positive for right, negative for left)
        """
        dir_to_angle = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
        
        from_angle = dir_to_angle[from_dir]
        to_angle = dir_to_angle[to_dir]
        
        # Calculate the difference
        angle_diff = to_angle - from_angle
        
        # Normalize to [-180, 180]
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        return angle_diff
    
    def generate_complete_sketch(self, commands):
        """
        Generate a complete Arduino sketch from the movement commands
        
        Args:
            commands: List of dictionaries with movement commands
            
        Returns:
            code: Complete Arduino sketch as a string
        """
        # Generate the executable function
        executable_function = self._generate_executable_function(commands)
        
        # Generate the complete sketch
        code = f"""// AUTO-GENERATED MAZE INSTRUCTIONS
#define M1A 9  // Motor 1 Pin A
#define M1B 10 // Motor 1 Pin B
#define M2A 11 // Motor 2 Pin A
#define M2B 12 // Motor 2 Pin B

void setup() {{
  Serial.begin(9600);
  pinMode(M1A, OUTPUT);
  pinMode(M1B, OUTPUT);
  pinMode(M2A, OUTPUT);
  pinMode(M2B, OUTPUT);
  
  // Initialize all motors to stopped
  digitalWrite(M1A, LOW);
  digitalWrite(M1B, LOW);
  digitalWrite(M2A, LOW);
  digitalWrite(M2B, LOW);
  
  Serial.println("Maze Solver Robot initialized.");
  Serial.println("Send 'S' to start maze execution.");
}}

void loop() {{
  if (Serial.available() > 0) {{
    char command = Serial.read();
    
    if (command == 'S' || command == 's') {{
      Serial.println("Starting maze execution...");
      executeMazePath();
      Serial.println("Maze execution completed!");
    }}
  }}
}}

// MOTOR CONTROL FUNCTIONS
void forward(int duration) {{
  digitalWrite(M1A, HIGH);
  digitalWrite(M1B, LOW);
  digitalWrite(M2A, HIGH);
  digitalWrite(M2B, LOW);
  delay(duration);
  stopMotors();
}}

void backward(int duration) {{
  digitalWrite(M1A, LOW);
  digitalWrite(M1B, HIGH);
  digitalWrite(M2A, LOW);
  digitalWrite(M2B, HIGH);
  delay(duration);
  stopMotors();
}}

void turnRight(int degrees) {{
  // Calculate turn duration based on degrees
  int turnDuration = degrees * 10;  // Approximate time for degree conversion
  
  digitalWrite(M1A, HIGH);
  digitalWrite(M1B, LOW);
  digitalWrite(M2A, LOW);
  digitalWrite(M2B, HIGH);
  delay(turnDuration);
  stopMotors();
}}

void turnLeft(int degrees) {{
  // Calculate turn duration based on degrees
  int turnDuration = degrees * 10;  // Approximate time for degree conversion
  
  digitalWrite(M1A, LOW);
  digitalWrite(M1B, HIGH);
  digitalWrite(M2A, HIGH);
  digitalWrite(M2B, LOW);
  delay(turnDuration);
  stopMotors();
}}

void stopMotors() {{
  digitalWrite(M1A, LOW);
  digitalWrite(M1B, LOW);
  digitalWrite(M2A, LOW);
  digitalWrite(M2B, LOW);
}}

{executable_function}
"""
        return code
    
    def _generate_executable_function(self, commands):
        """
        Generate the Arduino function that executes the movement commands
        
        Args:
            commands: List of dictionaries with movement commands
            
        Returns:
            function: String with the executeMazePath function
        """
        function = """// GENERATED MAZE PATH FUNCTION
void executeMazePath() {
  // Movement commands (F=forward, L/R=turn degrees, D=delay ms)
"""
        
        # Add each command to the function
        for cmd in commands:
            cmd_type = cmd['type']
            
            if cmd_type == 'F':
                function += f"  forward({cmd['duration']});  // F{cmd['duration']}\n"
            elif cmd_type == 'L':
                function += f"  turnLeft({cmd['angle']});  // L{cmd['angle']}\n"
            elif cmd_type == 'R':
                function += f"  turnRight({cmd['angle']});  // R{cmd['angle']}\n"
            elif cmd_type == 'D':
                function += f"  delay({cmd['duration']});  // D{cmd['duration']}\n"
        
        function += "}\n"
        return function


class EmbeddedCCodeGenerator:
    def __init__(self):
        """Initialize the Embedded C code generator"""
        pass
    
    def generate_complete_code(self, commands):
        """
        Generate complete Embedded C code from the movement commands
        
        Args:
            commands: List of dictionaries with movement commands
            
        Returns:
            code: Complete Embedded C code as a string
        """
        # Generate the executable function
        executable_function = self._generate_executable_function(commands)
        
        # Generate the complete C code
        c_code = """/* AUTO-GENERATED MAZE INSTRUCTIONS */
#include <stdint.h>
#include <stdbool.h>

/* Define motor pins */
#define M1A 9  /* Motor 1 Pin A */
#define M1B 10 /* Motor 1 Pin B */
#define M2A 11 /* Motor 2 Pin A */
#define M2B 12 /* Motor 2 Pin B */

/* Function prototypes */
void initializeMotors(void);
void forward(uint16_t duration);
void backward(uint16_t duration);
void turnRight(uint16_t degrees);
void turnLeft(uint16_t degrees);
void stopMotors(void);
void executeMazePath(void);
void delay_ms(uint16_t ms);
void digitalWrite(uint8_t pin, bool value);

/* Initialize UART if needed for debugging */
void initializeSerial(void) {
    /* Implementation would depend on specific microcontroller */
}

/* Initialize GPIO pins for motors */
void initializeMotors(void) {
    /* Set motor pins as outputs */
    /* Implementation would depend on specific microcontroller */
    
    /* Initialize all motors to stopped */
    digitalWrite(M1A, false);
    digitalWrite(M1B, false);
    digitalWrite(M2A, false);
    digitalWrite(M2B, false);
}

/* Main function */
int main(void) {
    /* Initialize hardware */
    initializeSerial();
    initializeMotors();
    
    /* Execute maze path */
    executeMazePath();
    
    /* Infinite loop */
    while(1) {
        /* Could implement command reception here */
    }
    
    return 0;
}

/* MOTOR CONTROL FUNCTIONS */
void forward(uint16_t duration) {
    digitalWrite(M1A, true);
    digitalWrite(M1B, false);
    digitalWrite(M2A, true);
    digitalWrite(M2B, false);
    delay_ms(duration);
    stopMotors();
}

void backward(uint16_t duration) {
    digitalWrite(M1A, false);
    digitalWrite(M1B, true);
    digitalWrite(M2A, false);
    digitalWrite(M2B, true);
    delay_ms(duration);
    stopMotors();
}

void turnRight(uint16_t degrees) {
    /* Calculate turn duration based on degrees */
    uint16_t turnDuration = degrees * 10;  /* Approximate time for degree conversion */
    
    digitalWrite(M1A, true);
    digitalWrite(M1B, false);
    digitalWrite(M2A, false);
    digitalWrite(M2B, true);
    delay_ms(turnDuration);
    stopMotors();
}

void turnLeft(uint16_t degrees) {
    /* Calculate turn duration based on degrees */
    uint16_t turnDuration = degrees * 10;  /* Approximate time for degree conversion */
    
    digitalWrite(M1A, false);
    digitalWrite(M1B, true);
    digitalWrite(M2A, true);
    digitalWrite(M2B, false);
    delay_ms(turnDuration);
    stopMotors();
}

void stopMotors(void) {
    digitalWrite(M1A, false);
    digitalWrite(M1B, false);
    digitalWrite(M2A, false);
    digitalWrite(M2B, false);
}

/* Placeholder for delay function - implementation depends on microcontroller */
void delay_ms(uint16_t ms) {
    /* Implementation would depend on specific microcontroller */
    /* For example, using a timer or cycle counting */
}

/* Placeholder for digitalWrite function - implementation depends on microcontroller */
void digitalWrite(uint8_t pin, bool value) {
    /* Implementation would depend on specific microcontroller */
    /* For example, setting GPIO registers directly */
}
"""
        
        # Return the combined code (C code + executable function)
        return c_code + executable_function
    
    def _generate_executable_function(self, commands):
        """
        Generate the C function that executes the movement commands
        
        Args:
            commands: List of dictionaries with movement commands
            
        Returns:
            function: String with the executeMazePath function
        """
        function = """/* GENERATED MAZE PATH FUNCTION */
void executeMazePath(void) {
  /* Movement commands (F=forward, L/R=turn degrees, D=delay ms) */
"""
        
        # Add each command to the function
        for cmd in commands:
            cmd_type = cmd['type']
            
            if cmd_type == 'F':
                function += f"  forward({cmd['duration']});  /* F{cmd['duration']} */\n"
            elif cmd_type == 'L':
                function += f"  turnLeft({cmd['angle']});  /* L{cmd['angle']} */\n"
            elif cmd_type == 'R':
                function += f"  turnRight({cmd['angle']});  /* R{cmd['angle']} */\n"
            elif cmd_type == 'D':
                function += f"  delay_ms({cmd['duration']});  /* D{cmd['duration']} */\n"
        
        function += "}\n"
        return function