#include <Arduino_RouterBridge.h>
#include <DHT.h>

#define DHT_PIN D8
#define DHT_TYPE DHT11

DHT dht(DHT_PIN, DHT_TYPE);

float temperatureC = NAN;
float humidityRH = NAN;

unsigned long lastRead = 0;
const unsigned long READ_INTERVAL = 2000;

float get_dht_temperature() {
  return temperatureC;
}

float get_dht_humidity() {
  return humidityRH;
}

void updateDHT11() {
  if (millis() - lastRead < READ_INTERVAL) {
    return;
  }

  lastRead = millis();

  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (!isnan(t) && t >= -40.0 && t <= 80.0) {
    temperatureC = t;
  }

  if (!isnan(h) && h >= 0.0 && h <= 100.0) {
    humidityRH = h;
  }
}

void setup() {
  Serial.begin(115200);

  dht.begin();

  if (!Bridge.begin()) {
    Serial.println("Bridge failed");
  }

  if (!Bridge.provide_safe(
        "get_dht_temperature",
        get_dht_temperature)) {
    Serial.println("Temperature RPC registration failed");
  }

  if (!Bridge.provide_safe(
        "get_dht_humidity",
        get_dht_humidity)) {
    Serial.println("Humidity RPC registration failed");
  }

  Serial.println("DHT11 Bridge ready");
}

void loop() {
  updateDHT11();
}