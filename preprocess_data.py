import pandas as pd

def run():
    #parse_dates will parse columns values into datetime objects.
    df = pd.read_csv('mittausdata.csv', sep=';', parse_dates=['timestamp'])

    # drop columns
    df = df.drop(['rssi2'], axis=1)

    # drop rows with measurements on channels other than 18
    df = df[df.channel == 18]
    df = df.drop(['channel'], axis=1)

    # convert sensorid and neigthbors values to node id by substracting 0x8100
    def sensorid_to_int(id_hex:str) -> int:
        return int(id_hex, base=16) - int('0x8100', base=16)

    df.sensorid = df.sensorid.apply(sensorid_to_int)
    df.neighbor = df.neighbor.apply(sensorid_to_int)

    # drop rows where neightbor is sensorid 0 as such node isn't specified in
    # material nor given reference location, making it useless for calculations
    df = df[df.neighbor != 0]

    # node 3 is marked as 'not used'
    df = df[df.sensorid != 3]
    df = df[df.neighbor != 3]

    print(df.info())

    assert sorted(df['sensorid'].unique()) == sorted(df['neighbor'].unique()), f'Error: set of sensorids and set of neighbors are not equal. This is fatal.'

    # save cleaned data to new file
    df.to_csv('prosessoitu_mittausdata.csv', index=False)

if __name__ == "__main__":
    run()