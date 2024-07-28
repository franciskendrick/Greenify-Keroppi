#include <Servo.h>

float eyesSmoothed = 90;
float eyesPrev = 90;

float lidsSmoothed = 180;
float lidsPrev = 180;

Servo eye_right;
Servo eye_left;
Servo eyelids;

void setup() {
  Serial.begin(9600);

  // Attach servos to pins
  eyelids.attach(9);
  eye_right.attach(10);
  eye_left.attach(11);

  // Move to start position initially
  eye_right.write(90);
  eye_left.write(90);
  eyelids.write(90);

  // Wait for 0.5 seconds
  delay(500); 
}

void loop() {
  MoveEyesTo(90);  // start

  MoveLidsTo(180);  // close
  MoveLidsTo(90);  // open

  delay(100);

  MoveEyesTo(55);  // left
  MoveEyesTo(125);  // right
  MoveEyesTo(90);  // start

  MoveLidsTo(180);  // close
  MoveLidsTo(90);  // open

  delay(100);
}

void MoveEyesTo(int targetPosition) {
  while (abs(eyesSmoothed - targetPosition) > 0.1) {
    eyesSmoothed = (targetPosition * 0.15) + (eyesPrev * 0.85);
    eyesPrev = eyesSmoothed;

    eye_right.write(int(eyesSmoothed));
    eye_left.write(int(eyesSmoothed));

    Serial.print("Target: ");
    Serial.print(targetPosition);
    Serial.print(" , Smoothed: ");
    Serial.println(eyesSmoothed);

    delay(10);
  }
}

void MoveLidsTo(int targetPosition) {
  while (abs(lidsSmoothed - targetPosition) > 0.1) {
    lidsSmoothed = (targetPosition * 0.2) + (lidsPrev * 0.8);
    lidsPrev = lidsSmoothed;

    eyelids.write(int(lidsSmoothed));

    Serial.print("Target: ");
    Serial.print(targetPosition);
    Serial.print(" , Smoothed: ");
    Serial.println(lidsSmoothed);

    delay(10);
  }
}
