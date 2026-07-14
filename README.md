## Research Question

Can radio frequency (RF) signal data alone — without cameras, radar, or GPS — be used to reliably detect the presence of a drone and classify which type it is?

## Expected Data Source(s)

DroneRF dataset (Allahham et al., 2019, Qatar University) — RF recordings from 3 drone models across operating modes (off, on/connected, hovering, flying, video recording), plus drone-free background RF noise. ~40+ GB, available via Mendeley Data: https://data.mendeley.com/datasets/f4c2b4n755/1 Links to an external site.
CardRF dataset (Medaiyese et al., 2022) — Outdoor RF recordings from UAV controllers, UAVs, Bluetooth, and Wi-Fi devices, captured at both line-of-sight and beyond-line-of-sight ranges (~65 GB). Available via IEEE DataPort: https://ieee-dataport.org/documents/cardinal-rf-cardrf-outdoor-uavuasdrone-rf-signals-bluetooth-and-wifi-signals-dataset Links to an external site.
Together these give labeled drone-present/drone-absent examples across multiple drone models, operating modes, and noise conditions — including the Bluetooth/Wi-Fi interference that any real-world detector would have to contend with in the shared 2.4 GHz band.

## Techniques

CNN on spectrograms — convert raw IQ signals to spectrograms via Short-Time Fourier Transform (STFT), then use a CNN to learn the frequency-domain "fingerprint" of each drone type.
LSTM / RNN — capture temporal patterns across operating modes (e.g., the signal looks different when a drone is hovering vs. actively flying).
Multi-class classification with SNR robustness testing — evaluate performance across varying signal-to-noise ratios to test how the model holds up under real-world interference, not just clean lab conditions.
Expected Results

A trained model that can, from RF signal data alone, (1) flag whether a drone is present in a given environment, and (2) classify which of the known drone types it is, with performance reported separately under clean and noisy/interference conditions to show how reliability degrades as real-world conditions get messier.

## Why This Question Is Important

I have worked in DoD for most of my career and off and on throughout my career, I've worked on project that focuses on drones.  Most existing counter-drone systems rely on cameras, radar, or GPS tracking and each of these has major blindspots (i.e. cameras don't do well at night or with weather challenges, radar struggles with small or slow-moving drones, and GPS-based tracking only works if the drone is broadcasting its own location and being honest about their location - typically they can be spoofed or not broadcast at all).  RF detection closes the gap on these blindspots because every drone has to communicate with its controller whether it wants to be tracked or be invisible.   

If this question goes unanswered, organization responsible for protecting airspace like airports, prisons, stadiums, and other critical infrastructure are left dependent on detection methods that can be evaded by GPS spoofing or GPS denied drones.Now that consumer drones have been cheap and available to common users, this is a real security gap that needs to be addressed.

The practical payoff to this project is a proof-of-concept showing that a relatively low-cost RF sensor paired with a trained model can give a security team an early, reliable warning that something is flying nearby and determine the possible device without needing someone to see it visually and without necessarily the drone's cooperation.  These events could trigger an alert, log an incident, and escalate to a human responder.
