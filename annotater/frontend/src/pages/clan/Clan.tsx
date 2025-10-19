import { useEffect, useState } from "react";
import { useProfileStore } from '../../contexts/profileStore';
import "./clan.css";

const BACKEND_URL = "http://localhost:8000/clan";
const COLONY_URL = "http://localhost:8000/colonies";

const Clan = () => {
  const { clan, id: userId, fetchProfileInfo } = useProfileStore();
  const [clans, setClans] = useState<any[]>([]);
  const [newClanName, setNewClanName] = useState("");
  const [newClanDescription, setNewClanDescription] = useState("");
  const [userClan, setUserClan] = useState<any>(null);
  const [memberProfiles, setMemberProfiles] = useState<any[]>([]);
  const [showTradeModal, setShowTradeModal] = useState(false);
  const [tradeTarget, setTradeTarget] = useState<any>(null);
  const [pendingTrades, setPendingTrades] = useState<any[]>([]);
  const [tradeOffer, setTradeOffer] = useState({
    offerResource: "food",
    offerAmount: 0,
    requestResource: "chitin",
    requestAmount: 0,
  });

  useEffect(() => {
    const initialize = async () => {
      await fetchProfileInfo();
      const updatedUserId = useProfileStore.getState().id;
      await fetchPendingTrades(updatedUserId);
      if (useProfileStore.getState().clan) {
        const clanId = useProfileStore.getState().clan;
        try {
          const response = await fetch(`${BACKEND_URL}/${clanId}`, { credentials: "include" });
          const data = await response.json();
          setUserClan(data);
          await fetchMemberProfiles(data.members);
        } catch (error) {
          console.error("Error fetching user's clan:", error);
        }
      }
      await fetchClans();
    };
    initialize();
  }, []);

  const fetchMemberProfiles = async (memberIds: string[]) => {
    try {
      const responses = await Promise.all(
        memberIds.map(id =>
          fetch(`http://localhost:8000/profiles/${id}`, { credentials: "include" })
            .then(res => res.ok ? res.json() : null)
        )
      );
      const validProfiles = responses.filter(profile => profile !== null);
      setMemberProfiles(validProfiles);
    } catch (error) {
      console.error("Error fetching member profiles:", error);
    }
  };

  const fetchClans = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/all`, { credentials: "include" });
      const data = await response.json();
      setClans(data);
    } catch (error) {
      console.error("Error fetching clans:", error);
    }
  };

  const handleCreateClan = async () => {
    if (!newClanName.trim()) return;
    try {
      const payload = {
        name: newClanName,
        max_size: 50,
        description: newClanDescription,
      };
      await fetch(`${BACKEND_URL}/create`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await fetchProfileInfo();
      const updatedClanId = useProfileStore.getState().clan;
      if (updatedClanId) {
        const response = await fetch(`${BACKEND_URL}/${updatedClanId}`, { credentials: "include" });
        const data = await response.json();
        setUserClan(data);
        await fetchMemberProfiles(data.members);
      }
      await fetchClans();
      setNewClanName("");
      setNewClanDescription("");
    } catch (error) {
      console.error("Error creating clan:", error);
    }
  };

  const handleJoinClan = async (id: string) => {
    try {
      await fetch(`${BACKEND_URL}/${id}/join`, { method: "POST", credentials: "include" });
      await fetchProfileInfo();
      const updatedClanId = useProfileStore.getState().clan;
      if (updatedClanId) {
        const response = await fetch(`${BACKEND_URL}/${updatedClanId}`, { credentials: "include" });
        const data = await response.json();
        setUserClan(data);
        await fetchMemberProfiles(data.members);
      }
      await fetchClans();
    } catch (error) {
      console.error("Error joining clan:", error);
    }
  };
  

  const handleLeaveClan = async (id: string) => {
    try {
      await fetch(`${BACKEND_URL}/${id}/leave`, {
        method: "POST",
        credentials: "include",
      });
      await fetchProfileInfo();
      setUserClan(null);
      setMemberProfiles([]);
      await fetchClans();
    } catch (error) {
      console.error("Error leaving clan:", error);
    }
  };

  const handleTradeSubmit = async () => {
    try {
      const res = await fetch(`${COLONY_URL}/${userId}`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch your colony info");
      const colonyData = await res.json();
  
      const resourceAmount = tradeOffer.offerResource === "food" ? colonyData.food : colonyData.chitin;
  
      if (tradeOffer.offerAmount > resourceAmount) {
        alert(`You do not have enough ${tradeOffer.offerResource} to offer this trade!`);
        return;
      }
  
      const payload = {
        from_user_id: userId,
        to_user_id: tradeTarget._id,
        offer_resource: tradeOffer.offerResource,
        offer_amount: tradeOffer.offerAmount,
        request_resource: tradeOffer.requestResource,
        request_amount: tradeOffer.requestAmount,
      };
  
      await fetch(`http://localhost:8000/trades/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
      });
  
      await fetchPendingTrades(userId);
  
      setShowTradeModal(false);
      setTradeOffer({ offerResource: "food", offerAmount: 0, requestResource: "chitin", requestAmount: 0 });
      setTradeTarget(null);
    } catch (error) {
      console.error("Error sending trade:", error);
    }
  };
  

  const fetchPendingTrades = async (userId: string) => {
    if (!userId) return;
    try {
      const res = await fetch(`http://localhost:8000/trades/pending/${userId}`, { credentials: "include" });
      if (res.ok) {
        const data = await res.json();
        setPendingTrades(data);
      }
    } catch (error) {
      console.error("Error fetching pending trades", error);
    }
  };
  

  const handleAcceptTrade = async (tradeId: string) => {
    try {
      await fetch(`http://localhost:8000/trades/accept/${tradeId}`, {
        method: "POST",
        credentials: "include",
      });
  
      await fetchProfileInfo();
      const updatedUserId = useProfileStore.getState().id;
      await fetchPendingTrades(updatedUserId);
  
      setPendingTrades((prev) => prev.filter((trade) => trade._id !== tradeId));
    } catch (error) {
      console.error("Error accepting trade:", error);
    }
  };
  
  const handleDeclineTrade = async (tradeId: string) => {
    try {
      await fetch(`http://localhost:8000/trades/decline/${tradeId}`, {
        method: "POST",
        credentials: "include",
      });
  
      const updatedUserId = useProfileStore.getState().id;
      await fetchPendingTrades(updatedUserId);
  
      setPendingTrades((prev) => prev.filter((trade) => trade._id !== tradeId));
    } catch (error) {
      console.error("Error declining trade:", error);
    }
  };

  return (
    <div className="clan-container">
      {userClan ? (
        <>
          <h1 className="clan-header">{userClan.name}</h1>
          <p className="clan-description" style={{ fontSize: "1rem", opacity: 0.8 }}>{userClan.description}</p>
          <div className="clan-section">
            <div className="clan-members">
              <h3 className="clan-header" style={{ color: '#3a2e1f' }}>Members:</h3>
              <div className="member-cards">
  {memberProfiles.map((member) => {
    const pendingTradeFromMember = pendingTrades.find(
      (trade) => trade.from_user_id === member._id
    );

    const hasTradeWithMember = pendingTrades.some(
      (trade) =>
        (trade.to_user_id === member._id && trade.from_user_id === userId) ||
        (trade.to_user_id === userId && trade.from_user_id === member._id)
    );

    return (
      <div key={member._id} className="member-card">
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <img src={member.picture ? member.picture : '/ant.png'} alt="Profile" className="member-avatar" />
          <strong>{member.name}</strong>
        </div>
        <p>{member._id === userClan.leader ? "Leader" : "Member"}</p>

        <div style={{ display: "flex", gap: "5px" }}>
          {userId === userClan.leader && member._id !== userClan.leader && (
            <button className="clan-kick-btn" onClick={() => console.log("Kick not implemented yet")}>
              Kick
            </button>
          )}

          {(member._id !== userId) && !hasTradeWithMember && (
            <button className="clan-trade-btn" onClick={() => { 
              setTradeTarget(member); 
              setShowTradeModal(true); 
            }}>
              Trade
            </button>
          )}

          {pendingTradeFromMember && member._id !== userId && (
            <div className="trade-actions">
              <p className="trade-header">Trade Request</p>
              <button
                className="trade-accept"
                onClick={() => handleAcceptTrade(pendingTradeFromMember._id)}
              >
                Accept
              </button>
              <button
                className="trade-decline"
                onClick={() => handleDeclineTrade(pendingTradeFromMember._id)}
              >
                Decline
              </button>
            </div>
          )}
        </div>
      </div>
    );
  })}
</div>

            </div>

            <button className="clan-btn" onClick={() => handleLeaveClan(userClan._id)} style={{ marginTop: "20px" }}>
              Leave Clan
            </button>
          </div>

          {showTradeModal && (
            <div className="modal-overlay">
              <div className="modal">
                <h2>Trade with {tradeTarget?.name}</h2>
                <div className="modal-form">
                  <label>Offer Resource:</label>
                  <select value={tradeOffer.offerResource} onChange={(e) => setTradeOffer({ ...tradeOffer, offerResource: e.target.value })}>
                    <option value="food">Food</option>
                    <option value="chitin">Chitin</option>
                  </select>
                  <input type="number" value={tradeOffer.offerAmount} onChange={(e) => setTradeOffer({ ...tradeOffer, offerAmount: parseInt(e.target.value) })} />

                  <label>Request Resource:</label>
                  <select value={tradeOffer.requestResource} onChange={(e) => setTradeOffer({ ...tradeOffer, requestResource: e.target.value })}>
                    <option value="food">Food</option>
                    <option value="chitin">Chitin</option>
                  </select>
                  <input type="number" value={tradeOffer.requestAmount} onChange={(e) => setTradeOffer({ ...tradeOffer, requestAmount: parseInt(e.target.value) })} />

                  <button className="clan-submit-trade" onClick={handleTradeSubmit}>Submit Trade</button>
                  <button className="clan-cancel-trade" onClick={() => setShowTradeModal(false)}>Cancel</button>
                </div>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          <h1 className="clan-header">Clans</h1>
          <p className="clan-description">
            Clans are great ways to interact with other players. You can trade resources, share strategies, and build a community.
          </p>

          <div className="clan-section">
            <h2 className="clan-subtitle">Create a Clan</h2>
            <div className="clan-custom-input">
              <input
                type="text"
                placeholder="Clan Name"
                className="clan-input"
                value={newClanName}
                onChange={(e) => setNewClanName(e.target.value)}
              />
            </div>
            <div className="clan-custom-input">
              <input
                type="text"
                placeholder="Clan Description"
                className="clan-input"
                value={newClanDescription}
                onChange={(e) => setNewClanDescription(e.target.value)}
              />
            </div>
            <button className="clan-btn" onClick={handleCreateClan}>Create Clan</button>
          </div>

          <div className="clan-section">
            <h2 className="clan-subtitle">Join an Existing Clan</h2>
            <div className="clan-list">
              {clans.length === 0 ? (
                <p className="clan-empty">No clans available. Be the first one!</p>
              ) : (
                clans.map((c) => (
                  <div key={c._id} className="clan-card">
                    <div className="clan-info">
                      <strong>{c.name}</strong> — {c.members.length} members
                      <p style={{ fontSize: "0.9rem", opacity: 0.7 }}>{c.description}</p>
                    </div>
                    <button className="clan-join-btn" onClick={() => handleJoinClan(c['_id'])}>
                      Join
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Clan;
