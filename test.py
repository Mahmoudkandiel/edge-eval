import sophon.sail as sail



if __name__ == '__main__':
    bmodel_path = "yolov8n_1684_f32.bmodel"
    engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
    graph_name = engine.get_graph_names()[0]
    print("graph_name:",graph_name)