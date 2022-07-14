def build_spec(scheme):
    spec = []
    layer_names = []
    for i in range(scheme["rows"]):
        spec.append([])
        for j in range(scheme["cols"]):
            spec[-1].append(None)
            for layer in tom_1_unet_def:
                if isinstance(tom_1_unet_def[layer], tuple):
                    if tom_1_unet_def[layer][0] == i + 1 and tom_1_unet_def[layer][1] == j + 1:
                        spec[-1].pop()
                        spec[-1].append({"rowspan": tom_1_unet_def[layer][2], "colspan": tom_1_unet_def[layer][3]})
                        layer_names.append(layer)
    return spec, layer_names


tom_1_unet_def = {
    # start row (y), start col (x), height, width
    'encoder0': (3, 1, 2, 2),
    'encoder1': (5, 2, 2, 2),
    'encoder2': (7, 3, 2, 2),
    'encoder3': (9, 4, 2, 2),
    'encoder4': (11, 5, 2, 2),
    'decoder0': (3, 12, 2, 2),
    'decoder1': (5, 11, 2, 2),
    'decoder2': (7, 10, 2, 2),
    'decoder3': (9, 9, 2, 2),
    'decoder4': (11, 8, 2, 2),
    'upsample1': (5, 15, 2, 2),
    'upsample2': (7, 15, 2, 2),
    'upsample3': (9, 15, 2, 2),
    'upsample4': (11, 15, 2, 2),
    'center': (13, 6, 2, 2),
    'final0': (1, 15, 2, 2),
    'final1': (1, 17, 2, 2),
    'final2': (1, 19, 2, 2),
    'raw': (1, 1, 2, 2),
    'rows': 15,
    "cols": 20,
}

tom_1_unet_def["spec"], tom_1_unet_def["names"] = build_spec(tom_1_unet_def)
