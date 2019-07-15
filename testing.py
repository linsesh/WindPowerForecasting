from batch_generator import PandasBatchGenerator

def keras_prediction_example(test_set, config ,model, n=200):
    test_generator = PandasBatchGenerator(test_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                          config["target_variable"], 1, config["skip_steps"])

    print("example of prediction :")
    for i in range(n):
        #print(test_set[test_generator.current_idx:test_generator.current_idx + test_generator.num_steps + test_generator.num_padding])
        inp, out = next(test_generator.generate())
        y = model.predict(inp, verbose=1)
        print(y)
        print(out)

def keras_batch_testing(test_set, model, config):
    test_generator = PandasBatchGenerator(test_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                          config["target_variable"], 1, config["skip_steps"])

    ev = model.evaluate_generator(test_generator.generate(),
                                  len(test_set) // ((config["batch_size"]*config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]),
                                  verbose=1)
    print("error on test set: [%s]" % ', '.join(map(str, ev)))
