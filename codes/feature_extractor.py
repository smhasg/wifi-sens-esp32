import pandas as pd


class FeatureExtractor:

    @staticmethod
    def packet_to_dataframe(packets):

        rows = []
        for p in packets:
            rows.append(
                [
                    p["rssi"],
                    p["channel"],
                    p["csi_len"],
                    *p["csi"][:128]
                ]

            )
        columns = [
            "rssi",
            "channel",
            "csi_len"
        ]

        columns += [

            f"cs{i}"

            for i in range(128)

        ]
        return pd.DataFrame(
            rows,
            columns=columns
        )



