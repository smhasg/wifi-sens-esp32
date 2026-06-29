#include <WiFi.h>
#include "esp_wifi.h"

void wifi_csi_callback(void *ctx, wifi_csi_info_t *data)
{
    Serial.printf("%lu,%d,%d,%d,",
                  millis(),
                  data->rx_ctrl.rssi,
                  data->rx_ctrl.channel,
                  data->len);

    for (int i = 0; i < data->len; i++)
    {
        Serial.print(data->buf[i]);

        if (i < data->len - 1)
            Serial.print(",");
    }

    Serial.println();
}

void setup()
{
    Serial.begin(921600);

    WiFi.mode(WIFI_STA);

    WiFi.begin("CSI_TEST", "12345678");

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
    }

    wifi_csi_config_t config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false
    };

    esp_wifi_set_promiscuous(true);

    esp_wifi_set_csi_rx_cb(wifi_csi_callback, NULL);

    esp_wifi_set_csi_config(&config);

    esp_wifi_set_csi(true);

    Serial.println("CSI STARTED");
}

void loop()
{
    delay(1000);
}