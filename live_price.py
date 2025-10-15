import websocket, json

def on_message(ws, message):
    data = json.loads(message)
    price = float(data['p'])
    print("BTC/USDT price:", price)

def on_open(ws):
    print("WebSocket connection opened.")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed.")

url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
ws = websocket.WebSocketApp(url,
                            on_message=on_message,
                            on_open=on_open,
                            on_error=on_error,
                            on_close=on_close)
ws.run_forever()
 