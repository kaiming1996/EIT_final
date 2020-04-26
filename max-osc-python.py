#!/usr/bin/env python
"""max-osc-python.py : demonstration of a service node communicating parameters and data with Max via OSC messaging.

Copyright (c) 2014-2017, Garth Zeglin.  All rights reserved.  Provided under the
terms of the BSD 3-clause license.

This uses txosc and Twisted to send and receive OSC messages.
"""

# references:
#   https://docs.scipy.org/doc/numpy/
#   https://bitbucket.org/arjan/txosc/wiki/Home
#   https://twistedmatrix.com/trac/wiki/Documentation

# standard Python modules
from __future__ import print_function
import time, argparse, random
import requests
import time
import datetime as dt
# NumPy system for numeric and matrix computation
import numpy as np

# Twisted networking framework
import twisted.internet.reactor
import twisted.internet.task
import twisted.internet.protocol

# TxOSC OpenSoundControl library
import txosc.osc
import txosc.dispatch
import txosc.async

# The associated Max patcher assumes that the Python node sends and receives using the following UDP port numbers:
PYTHON_NODE_RECV_PORT = 12001
PYTHON_NODE_SEND_PORT = 12000

################################################################
class OscServer(object):
    """The OscServer class holds all the application state: communication ports,
    message callback assignments, and dynamic parameters."""

    def __init__(self, recv_port = PYTHON_NODE_RECV_PORT, send_port = PYTHON_NODE_SEND_PORT, verbose = False):

        self.verbose = verbose
        self.recv_portnum = recv_port
        self.send_portnum = send_port
        self._reactor = None
        self._ping_count = 0
        
        # set default generator parameters
        self._reset_parameters()
        return

    def _reset_parameters(self):
        self._xfreq = 25
        self._yfreq = 25
        self._xphase = 0.0
        self._yphase = 0.0

    def getSensorData(self):
        PP_ADDRESS = "http://192.168.1.12"
        PP_CHANNELS = ["accX", "accY", "accZ"]
        url = PP_ADDRESS + "/get?" + ("&".join(PP_CHANNELS))
        data = requests.get(url=url).json()
        accX = data["buffer"][PP_CHANNELS[0]]["buffer"][0]
        accY = data["buffer"][PP_CHANNELS[1]]["buffer"][0]
        accZ = data["buffer"][PP_CHANNELS[2]]["buffer"][0]
        # print (accX, ' ', accY, ' ', accY)
        return [accX, accY, accZ]

    def listen( self, reactor ):
        """The listen method is called to establish the UDP ports to receive and send OSC messages."""
        self._reactor = reactor
        self.receiver = txosc.dispatch.Receiver()
        self._server_protocol = txosc.async.DatagramServerProtocol(self.receiver)
        self._server_port = reactor.listenUDP(self.recv_portnum, self._server_protocol, maxPacketSize=60000)
        if self.verbose: print( "Listening on osc.udp://localhost:%s", self.recv_portnum )

        self._client_protocol = txosc.async.DatagramClientProtocol()
        self._client_port = reactor.listenUDP(0, self._client_protocol, maxPacketSize=60000)
        if self.verbose: print( "Ready to send using %s" % self._client_port)

        # Set up the OSC message handling system.  As a convention for
        # legibility, the message callback methods have msg_ prepended to the
        # message, but this is not required.

        # Assign methods to receive messages intended for debugging the system.
        self.receiver.addCallback( "/reset", self.msg_reset )
        self.receiver.addCallback( "/quit",  self.msg_quit )
        self.receiver.addCallback( "/ping",  self.msg_ping )

        # Assign methods to receive parameter control messages.
        self.receiver.addCallback( "/xfreq", self.msg_xfreq)
        self.receiver.addCallback( "/yfreq", self.msg_yfreq)

        # Assign methods to receive other event functions.
        self.receiver.addCallback( "/nextframe", self.msg_nextframe)

        # Assign a default function to receive any other OSC message.
        self.receiver.fallback = self.msg_fallback

        return

    #### Message handlers. ############################################

    # Define a default handler for any unmatched message address.
    def msg_fallback(self, message, address):
        print("Received OSC message with unhandled address '%s' from %s: %s" % (message.address, address, message.getValues()))
        return

    def msg_reset( self, message, address):
        if self.verbose: print("Receive reset request.")
        self._reset_parameters()
        return

    def msg_quit( self, message, address):
        if self.verbose: print( "Received quit request, shutting down." )
        self._reactor.stop()

    def msg_ping( self, message, address):
        if self.verbose: print("Received ping request.")

        # reply to the IP address from which the message was received
        send_host = address[0]

        self._ping_count += 1
        self._client_protocol.send( txosc.osc.Message("/pong", self._ping_count), (send_host, self.send_portnum))
        return

    def msg_xfreq( self, message, address):
        self._xfreq = message.getValues()[0]
        if self.verbose: print("xfreq now %s" % self._xfreq)
        return

    def msg_yfreq( self, message, address):
        self._yfreq = message.getValues()[0]
        if self.verbose: print("yfreq now %s" % self._yfreq)
        return

    def msg_nextframe( self, message, address):
        if self.verbose: print("Generating next frame.")

        # create a two-channel trajectory signal as a 2xN array
        cols = 10
        rows = 2
        trajectory = np.ndarray((rows, cols), dtype=np.float32)

        # create index array for calculating functions
        tt = np.linspace(0.0, 1.0, cols, dtype=np.float32)
        
        # compute two sets of trajectory samples

        PP_ADDRESS = "http://192.168.1.12"
        PP_CHANNELS = ["accX", "accY", "accZ"]
        url = PP_ADDRESS + "/get?" + ("&".join(PP_CHANNELS))
        data = requests.get(url=url).json()
        accX = data["buffer"][PP_CHANNELS[0]]["buffer"][0]
        accY = data["buffer"][PP_CHANNELS[1]]["buffer"][0]
        accZ = data["buffer"][PP_CHANNELS[2]]["buffer"][0]
        
        trajectory[0,:] = accY
        trajectory[1,:] = accZ
        # trajectory[0,:] = 0.5 + 0.5*np.cos(4*tt + self._xphase)
        # trajectory[1,:] = 0.5 + 0.5*np.sin(4*tt + self._yphase)

        # self._xphase += 0.01 * (50 - self._xfreq)
        # self._yphase += 0.01 * (50 - self._yfreq)

        # Reformat the trajectory for sending as an OSC message.  The txosc Message
        # class doesn't recognize numpy values, so the pixel values are
        # converted to an ordinary Python list of numbers.  The transpose is
        # used to re-order the values so each plane is sent in succession.
        samples = [float(value) for value in trajectory.flat]
        msg = txosc.osc.Message("/trajectory", cols, rows, *samples)

        # send the trajectory back to the source of the request
        send_host = address[0]
        self._client_protocol.send( msg, (send_host, self.send_portnum))

        return


################################################################

# Script entry point.
if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = """Demonstration of a UDP OSC node which can communicate with Max.""")
    parser.add_argument( '--verbose', action='store_true', help = 'Emit debugging output.')
    args = parser.parse_args()

    # Set up the txosc UDP port listening for requests.
    osc_server = OscServer( verbose=args.verbose )
    osc_server.listen( twisted.internet.reactor )

    # Start the Twisted event loop.
    twisted.internet.reactor.run()

    if args.verbose: print("Event loop exited.")
