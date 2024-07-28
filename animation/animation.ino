#include <Servo.h>

float switch1Smoothed = 90; // Start at the middle position
float switch1Prev = 90;

Servo eye_right;
Servo eye_left;

void setup() {
  Serial.begin(9600);

  // Attach servos to pins
  eye_right.attach(10); // Change the pin number as needed
  eye_left.attach(11); // Change the pin number as needed

  // Move to start position initially
  eye_right.write(90);
  eye_left.write(90);
  delay(500); // Wait for 0.5 seconds
}

void loop() {
  smoothMoveTo(55);  // Look Left
  smoothMoveTo(125); // Look Right
  smoothMoveTo(90);  // Return to Start Position
}

void smoothMoveTo(int targetPosition) {
  while (abs(switch1Smoothed - targetPosition) > 0.1) {
    switch1Smoothed = (targetPosition * 0.15) + (switch1Prev * 0.85);
    switch1Prev = switch1Smoothed;

    eye_right.write(int(switch1Smoothed));
    eye_left.write(int(switch1Smoothed));

    Serial.print("Target: ");
    Serial.print(targetPosition);
    Serial.print(" , Smoothed: ");
    Serial.println(switch1Smoothed);

    delay(10); // Adjust the delay to control the smoothness
  }
}
