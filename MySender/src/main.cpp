#include <WiFi.h>

const char* ssid = "CSI_TEST";
const char* password = "12345678";

void setup() {

  Serial.begin(115200);

  WiFi.mode(WIFI_AP);

  WiFi.softAP(ssid, password, 1);

  Serial.println("Sender AP Started");
}

void loop() {


  delay(1000);
}