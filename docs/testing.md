# Testing

This document explains how to run the tests for the proxy.

## Unit Tests

The unit tests cover the core functionality of the proxy, including request/response conversion, routing, and streaming. To run the unit tests, use the following command:

```bash
make test
```

Alternatively, you can run the tests directly using `pytest`:

```bash
pytest
```

The tests are located in the `tests/` directory.

## Performance Tests

The performance tests are designed to measure the overhead of the proxy and the performance of the underlying models. To run the performance tests, use the following command:

```bash
python performance_test.py --model_id <model_id_from_models.yaml>
```

This script will send a series of requests to the proxy and measure the time it takes to receive a response. The results can be used to analyze the performance of the proxy and identify any bottlenecks.
