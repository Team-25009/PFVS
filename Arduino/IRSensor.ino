const int IRpin = 2;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(IRpin, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  
  if(digitalRead(IRpin)) {
    Serial.println("Sensing Object");
  }
  else {
    Serial.println("Nada");
  }
}

// bool check() {
//   if(digitalRead(IRpin)){
//     return true;
//   }
//   else{
//     return false;
//   }
// }
